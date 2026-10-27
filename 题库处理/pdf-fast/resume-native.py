"""Resume coordinate-only crops after the offline audit, retaining existing image vectors."""
import concurrent.futures
import fcntl
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import atomic_json

PDF=ROOT/'pdf-fast'
RUN=PDF/'crops-run'
INDEX=ROOT/'embedding-native-questions'
PYTHON=sys.executable


def supervise(command,log,env=None,producer=False):
    while True:
        with log.open('a') as output:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            failed=producer and json.loads((RUN/'status.json').read_text()).get('failed_papers',0)
            if producer and command[-1].endswith('native_sweep.py'):
                failed=bool(json.loads((PDF/'sweep-progress.json').read_text()).get('errors'))
        except OSError as error:
            if error.errno not in (2,116):raise
            failed=True
        if result.returncode==0 and not failed:return
        print('RETRY',command[-4:],result.returncode,flush=True)
        time.sleep(30)


def run(recheck=False):
    if recheck:
        if json.loads((RUN/'status.json').read_text())['state']!='finished':raise ValueError('initial crop batch must finish before repair')
    elif not (PDF/'audit-current.jsonl').exists():
        raise ValueError('Run pdf_native.py to audit cached PDFs before starting this batch')
    # Do not let old finished status files terminate consumers before the producer starts.
    for folder,prefix in [(RUN,'progress'),(INDEX,'status')]:
        for shard in range(8):
            path=folder/f'{prefix}-{shard}.json'
            state=json.loads(path.read_text()) if path.exists() else dict(shard=shard)
            atomic_json(path,dict(state,state='running',updated=time.time()))
    if not recheck:atomic_json(RUN/'status.json',dict(state='running',cropped_questions=0,processed_papers=0,failed_papers=0))
    atomic_json(RUN/'process.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),ocr=False,
                                      input='audit-current.jsonl',algorithm_revision='native-coordinates-v7',auto_restart=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        prefix='data-repair' if recheck else 'backfill'
        script='native_sweep.py' if recheck else 'native_crop.py'
        futures=[pool.submit(supervise,[PYTHON,'-u',str(ROOT/script)],RUN/(prefix+'.log'),producer=True)]
        for shard in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard))
            command=[PYTHON,'-u',str(ROOT/'embedding.py'),'build','--all','--shard',str(shard),'--follow','--native-crops']
            futures.append(pool.submit(supervise,command,INDEX/f'{prefix}-{shard}.log',env))
        futures.append(pool.submit(supervise,[PYTHON,'-u',str(ROOT/'embedding.py'),'merge','--follow','--native-crops'],INDEX/(prefix+'-publisher.log')))
        for future in concurrent.futures.as_completed(futures):future.result()
    atomic_json(PDF/'backfill-supervisor.json',dict(state='finished',pid=os.getpid(),updated=time.time(),data_repair=recheck))


if __name__=='__main__':
    for folder in (PDF,RUN,INDEX):folder.mkdir(parents=True,exist_ok=True)
    if '--check' in sys.argv:
        assert Path(PYTHON).is_file() and all((ROOT/name).is_file() for name in ['native_crop.py','native_sweep.py'])
        print('PASS current Python interpreter and batch entry points')
    else:
        with (PDF/'backfill-supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            atomic_json(PDF/'backfill-supervisor.json',dict(state='waiting_for_offline_audit',pid=os.getpid(),updated=time.time()))
            run(recheck='--data-repair' in sys.argv)
