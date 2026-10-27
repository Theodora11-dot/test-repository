"""Recheck every cached coordinate plan; regenerate only affected crops, without OCR."""
import collections
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import time
import sys

from common import ROOT,atomic_json,load_papers,question_skip_reason
from pdf_native import DIRECTORY,native_pages,find_answer_boundary
from native_crop import ALGORITHM,RECHECK_VERSION,DATA_RECHECK,RUN,crop_paper,plans


def recheck(paper,audit):
    path=RUN/'papers'/(audit['key']+'.json');old=json.loads(path.read_text())
    data_recheck=paper['subject']=='英语' and old.get('data_review_revision')!=DATA_RECHECK
    if old.get('coordinate_review_revision')==RECHECK_VERSION and not data_recheck and not any(question_skip_reason(r) for r in old['records']):
        return old,False
    pages=native_pages((ROOT/audit['pdf']).with_suffix('.html'))
    boundary=find_answer_boundary(pages);audit=dict(audit,answer_boundary=boundary)
    proposed,deferred=plans(paper,pages,boundary)
    old_keys=collections.Counter((r['page'],str(r['number'])) for r in old['records'])
    new_keys=collections.Counter((q['page'],str(q['number'])) for q in proposed)
    late_anchor=any(.92*p['height']<=s['box'][1]<p['body_bottom'] for i,p in enumerate(pages,1) for s in p['question_starts']
                    if i<boundary['page'] or (i==boundary['page'] and s['box'][1]<boundary['y']))
    oversized_line=any(abs(l.get('number_box',l['box'])[1]-l['box'][1])>8 for i,p in enumerate(pages,1) for l in p['lines']
                       if i<boundary['page'] or (i==boundary['page'] and l['box'][1]<boundary['y']))
    old_texts={(r['page'],str(r['number'])):''.join(r['text'].split()) for r in old['records']}
    text_changed=any(key in old_texts and old_texts[key]!=''.join(q['text'].split()) for q in proposed for key in [(q['page'],str(q['number']))])
    boundary_changed=any({k:v for k,v in r['native_answer_boundary'].items() if k!='source_y'}!=boundary for r in old['records'])
    regenerate=bool(new_keys-old_keys or text_changed or boundary_changed or late_anchor or oversized_line or any(question_skip_reason(r) for r in old['records'])
                    or any(v>1 for v in old_keys.values()) or any(v>1 for v in new_keys.values()))
    if regenerate:
        result=crop_paper(paper,audit,refresh=True)
    else:
        # No new rectangles are needed: remove old false section prompts and refresh their attribution.
        records=[r for r in old['records'] if (r['page'],str(r['number'])) in new_keys]
        kept={r['id'] for r in records}
        events=[dict(r,role='other',status='review',retired_by=RECHECK_VERSION) for r in old['records'] if r['id'] not in kept]
        result=dict(old,records=records,events=events,deferred=deferred,state='cropped' if records else 'review')
    result['coordinate_review_revision']=RECHECK_VERSION
    if paper['subject']=='英语':result['data_review_revision']=DATA_RECHECK
    result.pop('journaled_coordinate_review',None)
    atomic_json(path,result)
    return result,regenerate


def sweep():
    status_path=RUN/'status.json'
    source=json.loads(status_path.read_text())
    if source.get('state')!='finished':raise ValueError('initial crop producer is still running')
    papers={(p['subject'],p['id']):p for p in load_papers()};owners={}
    for shard in range(8):
        for p in json.loads((ROOT/'run'/f'shard-{shard}.json').read_text()):owners[(p['subject'],p['id'])]=shard
    audits=[json.loads(l) for l in (DIRECTORY/'audit-current.jsonl').open()]
    audits=[a for a in audits if a['state']=='checked']
    progress=dict(state='running',pid=os.getpid(),started=time.time(),planned_papers=len(audits),completed_papers=0,
                  regenerated_papers=0,cropped_questions=0,events=0,errors=[],revision=RECHECK_VERSION)
    states=[]
    for shard in range(8):
        path=RUN/f'progress-{shard}.json';state=json.loads(path.read_text());state.update(state='running',phase='coordinate_recheck',pid=os.getpid(),updated=time.time());atomic_json(path,state)
        states.append(dict(shard=shard,state='running',phase='coordinate_recheck',pid=os.getpid(),processed_papers=0,cropped_questions=0,failed_papers=0,review_papers=0))
    atomic_json(DIRECTORY/'sweep-progress.json',progress)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(recheck,papers[(a['subject'],a['paper_id'])],a):a for a in audits}
        for future in concurrent.futures.as_completed(futures):
            audit=futures[future];shard=owners[(audit['subject'],audit['paper_id'])];state=states[shard]
            try:
                result,regenerated=future.result();events=[] if result.get('journaled_coordinate_review')==RECHECK_VERSION else result.get('events',[])
                with (RUN/f'results-{shard}.jsonl').open('a') as journal:
                    for record in events:journal.write(json.dumps(record,ensure_ascii=False)+'\n')
                if result.get('journaled_coordinate_review')!=RECHECK_VERSION:
                    result['journaled_coordinate_review']=RECHECK_VERSION;result['journaled_revision']=ALGORITHM
                    atomic_json(RUN/'papers'/(audit['key']+'.json'),result)
                progress['completed_papers']+=1;progress['regenerated_papers']+=regenerated;progress['events']+=len(events)
                progress['cropped_questions']+=len(result['records']);state['processed_papers']+=1;state['cropped_questions']+=len(result['records']);state['review_papers']+=not bool(result['records'])
            except Exception as error:
                progress['errors'].append(dict(key=audit['key'],error=repr(error)));state['failed_papers']+=1
            progress['updated']=time.time()
            atomic_json(DIRECTORY/'sweep-progress.json',progress)
            print('COORDINATE_RECHECK',audit['key'],progress['completed_papers'],progress['regenerated_papers'],len(progress['errors']),flush=True)
    for shard,state in enumerate(states):
        state.update(state='finished',updated=time.time());atomic_json(RUN/f'progress-{shard}.json',state)
    if not progress['errors']:
        atomic_json(status_path,dict(state='finished',phase='coordinate_recheck',cropped_questions=progress['cropped_questions'],
                                   processed_papers=progress['completed_papers'],failed_papers=0,workers=states))
    progress.update(state='incomplete' if progress['errors'] else 'finished',updated=time.time())
    atomic_json(DIRECTORY/'sweep-progress.json',progress)
    return progress


def check():
    import tempfile,shutil
    from pathlib import Path
    from PIL import Image
    import native_crop as nc
    global ROOT,RUN
    original_root,original_run=ROOT,RUN
    saved=(nc.ROOT,nc.DIRECTORY,nc.RUN)
    papers={(p['subject'],p['id']):p for p in load_papers()}
    try:
        with tempfile.TemporaryDirectory() as temporary:
            ROOT=Path(temporary);RUN=ROOT/'pdf-fast/crops-run';nc.ROOT=ROOT;nc.DIRECTORY=ROOT/'pdf-fast';nc.RUN=RUN
            for key,number in [('math-155705','3'),('english-178541','1')]:
                original=original_run/'papers'/(key+'.json');raw=original.read_bytes();old=json.loads(raw)
                path=RUN/'papers'/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                for record in old['records']:
                    asset=ROOT/record['image'];asset.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original_root/record['image'],asset)
                    crop=ROOT/record['crop_file'];crop.parent.mkdir(parents=True,exist_ok=True);os.link(asset,crop)
                html=ROOT/'pdf-fast/pdfs'/(key+'.html');html.parent.mkdir(parents=True,exist_ok=True);html.symlink_to(original_root/'pdf-fast/pdfs'/(key+'.html'))
                audit=json.loads((original_root/'pdf-fast/papers'/(key+'.json')).read_text())
                result,_=recheck(papers['数学' if key.startswith('math') else '英语',key.split('-')[1]],audit)
                q=next(q for q in result['records'] if q['page']==1 and q['number']==number)
                with Image.open(ROOT/q['image']) as image:assert image.height>20
                assert not question_skip_reason(q)
                if key.startswith('english'):assert 'ChatGPT' not in q['text'] and 'I hope this email' not in q['text']
                assert original.read_bytes()==raw,'isolated recheck must not modify the live source'
    finally:
        ROOT,RUN=original_root,original_run;nc.ROOT,nc.DIRECTORY,nc.RUN=saved
    print('PASS isolated source recheck: recovered fraction question and clean writing prompt')


if __name__=='__main__' and '--check' in sys.argv:
    check()
elif __name__=='__main__':
    with (RUN/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(sweep(),ensure_ascii=False),flush=True)
