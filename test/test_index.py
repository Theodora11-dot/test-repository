"""Run without a GPU or model: python test/test_index.py."""
import tempfile
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / '题库处理')]
import index
import infer
import embedding
import prepare

from common import ROOT, pin_image, question_skip_reason
from index import connect, publish_index, store
from infer import rank
from embedding import unit_vector


class FakeEncoder:
    image_calls = 0
    device = 'cpu'

    def encode(self, texts=None, images=None):
        if images:
            self.image_calls += len(images)
        return np.ones((len(texts or images), 8), dtype=np.float32)


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / "original.jpg"
        original.write_bytes(b"image")
        asset = pin_image(original, "math-1-p001-page", root)
        db = connect(root / "index.sqlite3")
        record = dict(id="a", image=asset, fingerprint=[5, 1])
        store(db, record, [3, 4, 0])
        store(db, dict(id="b"), [0, 0, 1])
        store(db, record, [3, 4, 0])
        assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 2
        assert rank(db, [6, 8, 0], 1)[0]["id"] == "a"
        assert abs(rank(db, [6, 8, 0], 1)[0]["score"] - 1) < 1e-5
        original.rename(root / "moved.jpg")
        assert (root / rank(db, [3, 4, 0], 1)[0]["image"]).read_bytes() == b"image"
        (root / 'moved.jpg').write_bytes(b'updated')
        assert (root / asset).read_bytes() == b'updated'
        (root / 'moved.jpg').unlink()
        assert (root / asset).read_bytes() == b'updated'
        assert rank(db, [0, 0, 1], 1)[0]["id"] == "b"
        assert rank(db, [0, 0, 1], 0) == []
        for invalid in [[0, 0, 0], [np.nan, 1, 0], [[1, 2]]]:
            try:
                unit_vector(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid vector accepted")
        try:
            rank(db, [1, 0], 1)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong vector dimension accepted")
        store(db, dict(record, paper_key="a"), [3, 4, 0])
        store(db, dict(id="b", paper_key="b"), [0, 0, 1])
        for key, vector in [("a", [-1, 0, 0]), ("b", [1, 0, 0])]:
            db.execute("INSERT INTO papers VALUES (?,?,?)", (key, "{}", np.asarray(vector, dtype="<f2").tobytes()))
        db.commit()
        assert rank(db, [1, 0, 0], 2)[0]["id"] == "b", "metadata must affect ranking without excluding other results"
        assert len(rank(db, [1, 0, 0], 5)) == 2
        assert rank(db,[1,0,0],2)==rank(db,[1,0,0],2,corpus=infer.prepare_search(db)), 'batch loading must preserve exact ranking and scores'
        prepared = infer.prepare_search(db)
        reference = np.stack([np.frombuffer(row[0], dtype='<f2') for row in db.execute('SELECT vector FROM documents ORDER BY id')]).astype(np.float32)
        assert [r['id'] for r in prepared[0]] == ['a', 'b'] and np.array_equal(prepared[1], reference)
        store(db, dict(id='broken'), [1, 0])
        try:
            infer.prepare_search(db)
        except ValueError:
            pass
        else:
            raise AssertionError('inconsistent index vector lengths accepted')
        with db:
            db.execute("DELETE FROM documents WHERE id='broken'")
        tensor_results=rank(db,[1,0,0],2,corpus=infer.prepare_search(db,device='cpu'))
        numpy_results=rank(db,[1,0,0],2)
        assert [r['id'] for r in tensor_results]==[r['id'] for r in numpy_results]
        assert all(abs(a['score']-b['score'])<1e-5 for a,b in zip(tensor_results,numpy_results)), 'tensor scoring must agree with NumPy scoring'
        db.close()
    saved_root, saved_directory = index.ROOT, index.DIRECTORY
    try:
        with tempfile.TemporaryDirectory() as temporary:
            from PIL import Image
            root = Path(temporary)
            index.ROOT, index.DIRECTORY = root, root / "embedding-native-questions"
            index.DIRECTORY.mkdir()
            (root / "run").mkdir()
            (root / "assets").mkdir()
            base = dict(id="math-1-p001-q01", subject="数学", paper_id="1", page=1, role="question",
                        status="complete", number="1", title="单题", text="1. 求 x 的值", bbox=[0, 0, 10, 20])
            Image.new("RGB", (10, 20), "white").save(root / "assets" / (base["id"] + ".jpg"))
            Image.new("RGB", (10, 20), "white").save(root / "assets/math-1-p001-q06.jpg")
            db = connect(index.DIRECTORY / "index-0.sqlite3")
            metadata = db
            metadata.execute("INSERT INTO papers VALUES (?,?,?)", ("数学:1", json.dumps(dict(answer_ref=dict(boundary=dict(page=3)))),
                                                                     np.ones(8, dtype="<f2").tobytes()))
            metadata.commit()
            records = [base, dict(base, id="math-1-p001-q02", status="cross_page"),
                       dict(base, id="math-1-p001-q03", status="review"),
                       dict(base, id="math-1-p003-q01", page=3),
                       dict(base, id="math-1-p001-q04", role="answer"),
                       dict(base, id="math-1-p001-q05", text="【详解】故选 A"),
                       dict(base, id="english-1-p001-q01", subject="英语", text="1. A. good B. bad C. poor D. nice"),
                       dict(base, id="math-1-p001-page"), dict(base, id="math-1-p001-q06", bbox=[0, 0, 100, 100])]
            journal = root / "run/results-0.jsonl"
            journal.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records) + '{"id":')
            encoder = FakeEncoder()
            progress = index.build_questions(db, encoder, 0, source_directory=root/"run")
            assert encoder.image_calls == 1, "only the complete question crop may be encoded"
            assert (progress["excluded_answer_records"], progress["deferred_cross_page"], progress["deferred_review"],
                    progress["failed_questions"]) == (3, 2, 1, 2)
            offset = progress["offset"]
            progress = index.build_questions(db, encoder, 0, source_directory=root/"run")
            assert encoder.image_calls == 1 and progress["offset"] == offset, "resume must not re-encode or consume a partial JSON line"
            assert db.execute('SELECT count(*) FROM papers').fetchone()[0] == 1, 'native question builds must work without a legacy page database'
            r = json.loads(db.execute("SELECT record FROM documents").fetchone()[0])
            assert r["unit"] == "question" and r["id"].endswith("q01") and len(r["bbox"]) == 4
            with journal.open('r+b') as stream:
                stream.truncate(offset);stream.seek(offset)
                stream.write((json.dumps(dict(base,title='更新后的元数据'),ensure_ascii=False)+'\n').encode())
            index.build_questions(db,encoder,0,source_directory=root/"run")
            assert encoder.image_calls==1, 'metadata changes must reuse the existing image vector'
            assert json.loads(db.execute('SELECT record FROM documents').fetchone()[0])['title']=='更新后的元数据'
            assert question_skip_reason(dict(base, text="1. 第一题\n2. 第二题")) == "review"
            assert question_skip_reason(dict(base,text='1. 已知双曲线\n4\n3\n.\n求直线方程。')) is None, 'a denominator followed by punctuation on another line is not a question number'
            assert question_skip_reason(dict(base,text='2. 求直线方程。'))=='review', 'a different printed question number cannot stand in for the expected stem'
            assert question_skip_reason(dict(base, number="1-5")) == "review"
            assert question_skip_reason(dict(base, subject="英语", text="1. 本试卷共6页，考试时间90分钟")) == "review"
            assert question_skip_reason(dict(base, subject="英语", title="阅读理解", text="3. What did Louis do? A. Run. B. Rest.")) == "cross_page"
            assert question_skip_reason(dict(base, subject="英语", title="完形填空", text="A long passage " * 100)) == "cross_page"
            assert question_skip_reason(dict(base,subject='英语',title='完形填空',native_context_state='same_page_material',text='（9） amount of money that I use to feed myself.\n（6）A. good B. bad C. poor D. nice'))=='cross_page', 'a passage starting at an inline gap is incomplete even with a context label'
            form_pattern=next(p[2] for p in infer.QUESTION_PREFERENCES if p[0]=='动词形式填空')
            assert infer.re.search(form_pattern,'所给词的适当形式填空') and not infer.re.search(form_pattern,'选择合适的单词或短语并使用其适当形式填空'), 'a general word bank is not a supplied-verb exercise'
            with db:
                db.execute("INSERT INTO settings VALUES ('unit','question')")
            store(db, dict(base, id="math-1-p001-q99", role="answer"), [1]+[0]*7)
            assert [r["id"] for r in rank(db, np.ones(8), 5)] == [base["id"]], "question search must reject stale answer records"
            assert [r['id'] for r in rank(db,[1]+[0]*7,1,query_text='数学')]==[base['id']], 'an excluded high-score answer must not hide a valid lower-ranked question'
            assert question_skip_reason(dict(base,subject='英语',title='写作',text='1.邀请参加活动；'))=='review'
            assert question_skip_reason(dict(base,subject='英语',text='1. 假设你是李华，请写一封邮件，内容包括：'))=='review'
            assert question_skip_reason(dict(base,subject='英语',text='1. 请写一封信。\n【作者 ChatGPT】\nDear Jim, I hope ...'))=='answer_or_other'
            assert question_skip_reason(dict(base,text='1. 求x。\n解：由条件可得x=2。'))=='answer_or_other'
            assert question_skip_reason(dict(base,subject='英语',text='1. 请写一封信。\nOne possible version:\nDear Jim ...'))=='answer_or_other'
            face=dict(base,id='math-1-p001-q10',number='10',text='10. 如图，求直线AB与平面ACD的夹角。',image='assets/'+base['id']+'.jpg',paper_key='数学:1')
            line=dict(face,id='math-1-p001-q11',number='11',text='11. 如图，求直线AB与直线CD所成角。')
            duplicate=dict(face,id='math-1-p001-q12',number='12',text=face['text'].replace('10.','12.',1))
            store(db,face,[.98,.2]+[0]*6);store(db,line,[1]+[0]*7);store(db,duplicate,[.98,.2]+[0]*6)
            found=rank(db,[1]+[0]*7,5,query_text='找直线与平面的夹角 未知关键词')
            assert found[0]['id']==face['id'] and '线面角' in found[0]['matched_question_terms']
            assert line['id'] in [r['id'] for r in found], 'unmatched requirements must remain a soft preference'
            assert duplicate['id'] not in [r['id'] for r in found], 'another compilation must not repeat the same question'
            assert rank(db,[1]+[0]*7,1,query_text='不存在的要求')[0]['id']==line['id'], 'all-unmatched preferences must fall back to vector ranking'
            committed_offset=json.loads(db.execute("SELECT value FROM settings WHERE key='crop_progress'").fetchone()[0])['offset']
            for number in (20,21):
                record=dict(base,id=f'math-1-p001-q{number}',number=str(number),text=f'{number}. 求x。')
                Image.new('RGB',(10,20),'white').save(root/'assets'/(record['id']+'.jpg'))
                with journal.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            class InterruptedEncoder(FakeEncoder):
                def encode(self,texts=None,images=None):
                    if images and self.image_calls==1:raise KeyboardInterrupt('simulated worker termination')
                    return super().encode(texts,images)
            saved_monotonic=index.time.monotonic;index.time.monotonic=lambda:1
            try:
                try:index.build_questions(db,InterruptedEncoder(),0,source_directory=root/"run")
                except KeyboardInterrupt:pass
                else:raise AssertionError('interruption was not exercised')
            finally:index.time.monotonic=saved_monotonic
            assert db.execute("SELECT count(*) FROM documents WHERE id='math-1-p001-q20'").fetchone()[0]==0
            assert json.loads(db.execute("SELECT value FROM settings WHERE key='crop_progress'").fetchone()[0])['offset']==committed_offset, 'vectors and resume offset must roll back together'
            resumed=FakeEncoder();index.build_questions(db,resumed,0,source_directory=root/"run")
            assert resumed.image_calls==2 and db.execute("SELECT count(*) FROM documents WHERE id IN ('math-1-p001-q20','math-1-p001-q21')").fetchone()[0]==2
            fresh=dict(base,id='math-2-p001-q01',paper_id='2')
            Image.new('RGB',(10,20),'white').save(root/'assets'/(fresh['id']+'.jpg'))
            with journal.open('a') as stream:stream.write(json.dumps(fresh,ensure_ascii=False)+'\n')
            standalone=FakeEncoder();index.build_questions(db,standalone,0,source_directory=root/"run")
            assert standalone.image_calls==1 and db.execute("SELECT count(*) FROM papers WHERE id='数学:2'").fetchone()[0]==1, 'a new paper must generate its title vector without any legacy database'
            db.close()
    finally:
        index.ROOT, index.DIRECTORY = saved_root, saved_directory
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = connect(root / "index.sqlite3")
        store(original, dict(id="old"), [1, 0, 0])
        original.close()
        reader = sqlite3.connect((root / "index.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
        assert reader.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
        for shard in range(8):
            db = connect(root / f"index-{shard}.sqlite3")
            with db: db.execute("INSERT INTO settings VALUES ('encoder','test')")
            store(db, dict(id=str(shard)), [1, 0, 0])
            db.close()
        assert publish_index(root)["indexed_question_pages"] == 8
        assert reader.execute("SELECT count(*) FROM documents").fetchone()[0] == 1, "open readers must retain their old snapshot"
        fresh = sqlite3.connect(root / "index.sqlite3")
        assert fresh.execute("SELECT count(*) FROM documents").fetchone()[0] == 8
        fresh.close()
        reader.close()
    print("PASS: ranking, validation, idempotency, move-safe images, standalone question builds, checkpoint recovery and atomic publication")


def check_inference():
    saved_model = embedding.MODEL
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding.MODEL = root/'Qwen3-VL-Embedding-2B'
            embedding.MODEL.mkdir()
            (embedding.MODEL/'download-manifest.json').write_text(json.dumps([dict(Path='model.safetensors', Sha256='0'*64)]))
            path = root/'index.sqlite3'
            db = connect(path)
            embedding.check_encoder(db, initialize=True)
            record = dict(id='math-1-p001-q01', subject='数学', paper_id='1', page=1, role='question',
                          status='complete', number='1', title='数学练习', text='1. 求函数值。', image='native-assets/q.jpg')
            store(db, record, np.ones(8))
            db.close()
            before = path.read_bytes()
            db = infer.open_index(path)
            try:
                embedding.check_encoder(db)
                found = infer.search(db, FakeEncoder(), '找数学题 未知关键词', corpus=infer.prepare_search(db))
                assert len(found) == 1 and found[0]['image'] == record['image']
                assert infer.search(db, FakeEncoder(), '  ') == []
                for query, limit in [('太长'*3000, 5), ('数学', 0)]:
                    try:
                        infer.search(db, FakeEncoder(), query, limit)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('invalid inference input accepted')
                try:
                    embedding.check_encoder(db, max_pixels=4096)
                except ValueError:
                    pass
                else:
                    raise AssertionError('encoder mismatch accepted')
                try:
                    db.execute('DELETE FROM documents')
                except sqlite3.OperationalError:
                    pass
                else:
                    raise AssertionError('inference database must be read-only')
            finally:
                db.close()
            assert path.read_bytes() == before and not path.with_name('index.sqlite3-wal').exists()
            missing = root/'missing.sqlite3'
            try:
                infer.open_index(missing)
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError('a missing index must not be created')
            assert not missing.exists()
    finally:
        embedding.MODEL = saved_model
    subprocess.run([sys.executable, '-c',
                    "import infer,sys; assert all(name not in sys.modules for name in ['index','prepare','eval','pdf_native','native_crop','pipeline','torch'])"],
                   cwd=ROOT.parent, check=True)
    assert Path(infer.__file__).resolve() == ROOT.parent/'infer.py'
    assert infer.INDEX == ROOT/'embedding-native-questions/index.sqlite3'
    entry = str(ROOT.parent/'infer.py')
    help_result = subprocess.run([sys.executable, entry, '--help'], cwd=tempfile.gettempdir(),
                                 capture_output=True, text=True, check=True)
    assert '--index' in help_result.stdout
    empty_result = subprocess.run([sys.executable, entry, '  '], cwd=tempfile.gettempdir(),
                                  capture_output=True, text=True, check=True)
    assert json.loads(empty_result.stdout) == []
    print('PASS standalone inference, tolerant queries, input/model checks and read-only indexes')


def check_preparation():
    saved = prepare.ROOT, prepare.PDF, prepare.RUN, prepare.INDEX, prepare.supervise
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare.ROOT, prepare.PDF = root, root/'pdf-fast'
            prepare.RUN, prepare.INDEX = root/'pdf-fast/crops-run', root/'embedding-native-questions'
            prepare.RUN.mkdir(parents=True)
            prepare.INDEX.mkdir()
            (prepare.PDF/'audit-current.jsonl').write_text('')
            calls = []
            prepare.supervise = lambda command, log, env=None, producer=False: calls.append((command, env, producer))
            prepare.run()
            producers = [command for command, _, producer in calls if producer]
            workers = [(command, env) for command, env, producer in calls if 'build' in command]
            assert len(calls) == 10 and len(producers) == 1
            assert producers[0][-1] == str(root/'native_crop.py')
            assert {env['CUDA_VISIBLE_DEVICES'] for _, env in workers} == {str(i) for i in range(8)}
            assert all(command[2] == str(root/'index.py') for command, _ in workers)
            assert len([command for command, _, _ in calls if 'merge' in command]) == 1
            assert json.loads((prepare.PDF/'backfill-supervisor.json').read_text())['state'] == 'finished'
    finally:
        prepare.ROOT, prepare.PDF, prepare.RUN, prepare.INDEX, prepare.supervise = saved
    print('PASS isolated preparation starts one crop producer, eight GPU index workers and one publisher')


if __name__ == "__main__":
    check()
    check_inference()
    check_preparation()
