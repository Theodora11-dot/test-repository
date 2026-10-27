"""Compare every current coordinate crop against the native vector shards; never OCR."""
import argparse
import collections
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pipeline import ROOT,atomic_json
from native_crop import ALGORITHM,RUN,RECHECK_VERSION,DATA_RECHECK
from pdf_native import DIRECTORY
from embedding import question_skip_reason

INDEX=ROOT/'embedding-native-questions'


def vector_ok(blob):
    if len(blob)!=2048*2:return False
    vector=np.frombuffer(blob,dtype='<f2').astype(np.float32)
    return bool(np.isfinite(vector).all() and abs(float(np.linalg.norm(vector))-1)<.005)


def record_ok(record,expected):
    return record.get('unit')=='question' and record.get('encoding')=='image_only' and all(record.get(k)==v for k,v in expected.items())


def image_error(path,box,fingerprint):
    for attempt in range(3):
        try:
            stat=path.stat()
            if [stat.st_size,stat.st_mtime_ns]!=fingerprint:return 'image changed since encoding'
            with Image.open(path) as image:
                if image.size!=(box[2]-box[0],box[3]-box[1]):return 'image dimensions differ from crop'
            return None
        except OSError as error:
            if error.errno not in (2,116) or attempt==2:return repr(error)
            time.sleep(.2)


def audit(repair=False):
    audits=[json.loads(line) for line in (DIRECTORY/'audit-current.jsonl').open()]
    owners={};english_sources=set()
    for shard in range(8):
        for paper in json.loads((ROOT/'run'/f'shard-{shard}.json').read_text()):
            key=('math' if paper['subject']=='数学' else 'english')+'-'+paper['id'];owners[key]=shard
            if paper['subject']=='英语':english_sources.add(key)
    expected={};incomplete=[];deferred=collections.Counter();papers=0;invalid=[]
    for source in audits:
        if source['state']!='checked':continue
        path=RUN/'papers'/(source['key']+'.json')
        result=json.loads(path.read_text()) if path.exists() else {}
        data_pending=source['key'] in english_sources and result.get('data_review_revision')!=DATA_RECHECK
        if result.get('algorithm_revision')!=ALGORITHM or result.get('source_pages')!=source['source_pages'] or result.get('coordinate_review_revision')!=RECHECK_VERSION or data_pending:
            incomplete.append(source['key']);continue
        papers+=1;deferred.update(r['reason'] for r in result.get('deferred',[]))
        for record in result['records']:
            if record['id'] in expected:raise ValueError('duplicate crop ID: '+record['id'])
            expected[record['id']]=(owners[source['key']],record)
            reason=question_skip_reason(record)
            if reason:invalid.append(dict(id=record['id'],reason=reason))
    if repair and incomplete:raise ValueError('crop producer is incomplete; repair refused')
    if repair and invalid:raise ValueError('source crops contain excluded or incomplete records; fix the crops before vector repair')
    seen=set();extras=[];stale=[];bad_vectors=[];fingerprints={};digests={};queued=[[] for _ in range(8)]
    print('COVERAGE_SOURCES',papers,len(expected),'invalid',len(invalid),flush=True)
    with contextlib.ExitStack() as stack:
        if repair:
            # These are the same locks held by the consumers and publisher, including across NFS clients.
            for name in [f'process-{i}.lock' for i in range(8)]+['publisher.lock']:
                lock=stack.enter_context((INDEX/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for shard in range(8):
            path=INDEX/f'index-{shard}.sqlite3'
            db=sqlite3.connect(path if repair else path.resolve().as_uri()+'?mode=ro',uri=not repair,timeout=60)
            with contextlib.closing(db):
                delete=[]
                for qid,raw,blob in db.execute('SELECT id,record,vector FROM documents'):
                    target=expected.get(qid)
                    if target is None or target[0]!=shard:
                        extras.append(qid);delete.append(qid);continue
                    seen.add(qid);record=json.loads(raw)
                    fingerprints[qid]=record.get('fingerprint')
                    digests[qid]=hashlib.sha256(blob).digest()
                    if not record_ok(record,target[1]):stale.append(qid);queued[shard].append(target[1])
                    if not vector_ok(blob):bad_vectors.append(qid);delete.append(qid);queued[shard].append(target[1])
                if repair:
                    with db:db.executemany('DELETE FROM documents WHERE id=?',((qid,) for qid in delete))
            print('COVERAGE_SHARD',shard,len(seen),'extra',len(extras),'stale',len(stale),flush=True)
        missing=sorted(set(expected)-seen)
        for qid in missing:
            shard,record=expected[qid];queued[shard].append(record)
        if repair:
            for shard,records in enumerate(queued):
                with (RUN/f'results-{shard}.jsonl').open('a') as journal:
                    for record in {r['id']:r for r in records}.values():journal.write(json.dumps(record,ensure_ascii=False)+'\n')
    report=dict(updated=time.time(),algorithm_revision=ALGORITHM,audited_papers=len(audits),current_crop_papers=papers,
                pending_sources=[r['key'] for r in audits if r['state']!='checked'],incomplete_crop_papers=incomplete,
                expected_crops=len(expected),indexed_current_crops=len(seen),extra_ids=extras,missing_ids=missing,
                stale_record_ids=stale,bad_vector_ids=bad_vectors,deferred=dict(deferred),repair_applied=repair,
                invalid_source_records=invalid,
                queued_records=sum(len({r['id'] for r in records}) for records in queued),
                coverage_complete=not(incomplete or extras or missing or stale or bad_vectors or invalid))
    published=set();published_extra=[];published_stale=[];published_vectors=[]
    with contextlib.closing(sqlite3.connect((INDEX/'index.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)) as snapshot:
        for qid,raw,blob in snapshot.execute('SELECT id,record,vector FROM documents'):
            published.add(qid)
            if qid not in expected:published_extra.append(qid);continue
            record=json.loads(raw)
            if not record_ok(record,expected[qid][1]) or record.get('fingerprint')!=fingerprints.get(qid):published_stale.append(qid)
            if hashlib.sha256(blob).digest()!=digests.get(qid):published_vectors.append(qid)
    report.update(published_crops=len(published),published_extra_ids=published_extra,published_missing_ids=sorted(set(expected)-published),
                  published_stale_ids=published_stale,published_bad_vector_ids=published_vectors)
    report['coverage_complete'] &= not(published_extra or report['published_missing_ids'] or published_stale or published_vectors)
    print('COVERAGE_PUBLISHED',len(published),'images_to_verify',len(seen),flush=True)
    def check_image(qid):
        record=expected[qid][1]
        return qid,image_error(ROOT/record['image'],record['bbox'],fingerprints[qid])
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        report['bad_images']=[]
        for count,(qid,error) in enumerate(pool.map(check_image,sorted(seen)),1):
            if error:report['bad_images'].append(dict(id=qid,error=error))
            if count%10000==0:print('COVERAGE_IMAGES',count,len(seen),'bad',len(report['bad_images']),flush=True)
    report['verified_images']=len(seen)-len(report['bad_images'])
    report['coverage_complete'] &= not report['bad_images']
    report['finished']=time.time()
    atomic_json(DIRECTORY/'vector-coverage.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repair',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        expected=dict(id='q1',role='question',version='v2',status='complete')
        stored=dict(expected,unit='question',encoding='image_only',grade='高一')
        assert record_ok(stored,expected) and not record_ok(dict(stored,version='v1'),expected) and not record_ok(dict(stored,role='answer'),expected) and not record_ok(dict(stored,encoding='image_and_text'),expected)
        good=(np.ones(2048,dtype=np.float32)/np.sqrt(2048)).astype('<f2').tobytes()
        assert vector_ok(good) and not vector_ok(b'\x00'*4096) and not vector_ok(np.full(2048,np.nan,dtype='<f2').tobytes()) and not vector_ok(good[:-2])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'q.jpg';Image.new('RGB',(8,12),'white').save(path);stat=path.stat();fp=[stat.st_size,stat.st_mtime_ns]
            assert image_error(path,[30,40,38,52],fp) is None
            assert image_error(path,[30,40,39,52],fp)=='image dimensions differ from crop'
            assert image_error(path,[30,40,38,52],[0,0])=='image changed since encoding'
        print('PASS vector dimensions, finite values and normalization')
    else:
        result=audit(args.repair);print(json.dumps({k:v for k,v in result.items() if not k.endswith('_ids') and k not in ['incomplete_crop_papers','deferred']},ensure_ascii=False),flush=True)
