"""Prepare question data: audit PDFs, crop questions and build/publish vectors on eight GPUs."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, atomic_json

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
        raise ValueError('Run prepare.py audit to audit cached PDFs before starting this batch')
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
            command=[PYTHON,'-u',str(ROOT/'index.py'),'build','--shard',str(shard),'--follow']
            futures.append(pool.submit(supervise,command,INDEX/f'{prefix}-{shard}.log',env))
        futures.append(pool.submit(supervise,[PYTHON,'-u',str(ROOT/'index.py'),'merge','--follow'],INDEX/(prefix+'-publisher.log')))
        for future in concurrent.futures.as_completed(futures):future.result()
    atomic_json(PDF/'backfill-supervisor.json',dict(state='finished',pid=os.getpid(),updated=time.time(),data_repair=recheck))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['audit', 'build', 'repair', 'status'])
    parser.add_argument('--auth-file', type=Path, help='audit：可选的现有网站会话凭证；省略时仅检查缓存')
    parser.add_argument('--sample', type=int, default=0, help='audit：仅检查指定数量的试卷，0 表示全部')
    parser.add_argument('--check', action='store_true', help='仅检查批处理入口，不启动任务')
    args = parser.parse_args()
    if args.check:
        assert Path(PYTHON).is_file() and all((ROOT/name).is_file() for name in ['pdf_native.py','native_crop.py','native_sweep.py','index.py'])
        print('PASS current Python interpreter and preparation entry points')
        return
    if args.command is None:
        parser.error('请指定 audit、build、repair 或 status')
    if args.sample < 0 or (args.command != 'audit' and (args.auth_file or args.sample)):
        parser.error('--auth-file 和非负的 --sample 仅用于 audit')
    if args.command == 'audit':
        from pdf_native import audit
        audit(args.auth_file, args.sample)
        return
    if args.command == 'status':
        subprocess.run([PYTHON, str(ROOT/'index.py'), 'status'], check=True)
        return
    for folder in (PDF,RUN,INDEX):
        folder.mkdir(parents=True,exist_ok=True)
    with (PDF/'backfill-supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(PDF/'backfill-supervisor.json',dict(state='waiting_for_offline_audit',pid=os.getpid(),updated=time.time()))
        run(recheck=args.command == 'repair')


if __name__ == '__main__':
    main()
