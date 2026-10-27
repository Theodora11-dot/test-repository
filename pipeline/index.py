"""Build and atomically publish native single-question vector indexes."""
import argparse
import contextlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from common import ROOT, atomic_json, question_skip_reason
from embedding import Encoder, MAX_PIXELS, check_encoder, unit_vector

DIRECTORY = ROOT / 'embedding-native-questions'


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, record TEXT NOT NULL, vector BLOB NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, record TEXT NOT NULL, vector BLOB NOT NULL)")
    db.commit()
    return db

def publish_index(directory=DIRECTORY):
    """One publisher builds and atomically replaces a closed, read-only query snapshot."""
    temporary = directory / f".index-{os.getpid()}-{time.time_ns()}.sqlite3"
    # ponytail: SQLite builds on the node's local disk; one sequential NFS copy avoids random network writes.
    scratch=tempfile.TemporaryDirectory(prefix='question-index-')
    local=Path(scratch.name)/'index.sqlite3'
    destination = connect(local)
    try:
        expected = None
        for shard in range(8):
            path = directory / f"index-{shard}.sqlite3"
            if not path.exists():
                raise ValueError("缺少分片索引: " + str(path))
            source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
            try:
                source.execute("BEGIN")
                settings = source.execute("SELECT value FROM settings WHERE key='encoder'").fetchone()
                if expected is not None and settings != expected:
                    raise ValueError("分片编码设置不一致")
                expected = settings
                with destination:
                    for table, columns in [("documents", 3), ("papers", 3), ("settings", 2)]:
                        cursor = source.execute("SELECT * FROM " + table)
                        for rows in iter(lambda: cursor.fetchmany(500), []):
                            if table == "documents" and directory.name == "embedding-native-questions":
                                rows = [row for row in rows if question_skip_reason(json.loads(row[1])) is None]
                            destination.executemany("INSERT OR REPLACE INTO " + table + " VALUES (" +
                                                    ",".join("?" for _ in range(columns)) + ")", rows)
            finally:
                source.close()
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("发布前索引完整性检查失败")
        unit = destination.execute("SELECT value FROM settings WHERE key='unit'").fetchone()
        count_key = "indexed_question_crops" if unit == ("question",) else "indexed_question_pages"
        state = dict(updated=time.time(), **{count_key: destination.execute("SELECT count(*) FROM documents").fetchone()[0]},
                     papers=destination.execute("SELECT count(*) FROM papers").fetchone()[0])
        destination.close()
        with local.open('rb') as source,temporary.open('wb') as output:
            shutil.copyfileobj(source,output,16*1024*1024)
            output.flush();os.fsync(output.fileno())
        temporary.replace(directory / "index.sqlite3")
        atomic_json(directory / "published.json", state)
        return state
    finally:
        destination.close()
        temporary.unlink(missing_ok=True)
        scratch.cleanup()

def store(db, record, vector,commit=True):
    blob = unit_vector(vector).astype("<f2").tobytes()
    with db if commit else contextlib.nullcontext():
        db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?)",
                   (record["id"], json.dumps(record, ensure_ascii=False), blob))

def build_questions(db, encoder, shard, follow=False, source_directory=None):
    """Consume the existing crop journal; never OCR or encode a source page here."""
    from PIL import Image
    source_directory = source_directory or ROOT / "pdf-fast/crops-run"
    journal = source_directory / f"results-{shard}.jsonl"
    progress_path = DIRECTORY / f"status-{shard}.json"
    saved = db.execute("SELECT value FROM settings WHERE key='crop_progress'").fetchone()
    progress = json.loads(saved[0]) if saved and json.loads(saved[0]).get("version") == "questions-v1" else dict(
        version="questions-v1", offset=0, seen_records=0, completed_questions=0,
        skipped_questions=0, excluded_answer_records=0, deferred_cross_page=0, deferred_review=0, failed_questions=0)
    progress.update(started=time.time(), pid=os.getpid(), shard=shard, state="running",checkpoint_policy='32_records_or_5_seconds')
    checkpoint_seen=progress['seen_records'];checkpoint_at=time.monotonic()
    atomic_json(progress_path,progress)
    try:
        while True:
            source_status = source_directory / f"progress-{shard}.json"
            try:
                finished = source_status.exists() and json.loads(source_status.read_text()).get("state") == "finished"
            except OSError as error:
                if error.errno not in (2, 116):
                    raise
                time.sleep(1)
                continue
            if journal.exists():
                if journal.stat().st_size < progress["offset"]:
                    raise ValueError("裁剪记录被截短，保留断点等待检查: " + str(journal))
                with journal.open("rb") as source:
                    source.seek(progress["offset"])
                    while True:
                        line = source.readline()
                        if not line.endswith(b"\n"):
                            break  # Producer may still be writing this record.
                        record = {}
                        try:
                            record = json.loads(line)
                            qid = record["id"]
                            prefix = ("math" if record["subject"] == "数学" else "english") + "-" + record["paper_id"]
                            if not re.fullmatch(re.escape(prefix) + r"-p\d+-q\d+", qid):
                                raise ValueError("只接受逐题裁剪 ID: " + qid)
                            key = record["subject"] + ":" + record["paper_id"]
                            # ponytail: cache one title vector per paper in this shard.
                            row = db.execute("SELECT record,vector FROM papers WHERE id=?", (key,)).fetchone()
                            paper_record = json.loads(row[0]) if row else None
                            reason = question_skip_reason(record, paper_record)
                            if reason:
                                db.execute("DELETE FROM documents WHERE id=?", (qid,))
                                counter = {"answer_or_other": "excluded_answer_records", "cross_page": "deferred_cross_page",
                                           "review": "deferred_review"}[reason]
                                progress[counter] += 1
                                with (DIRECTORY / f"deferred-{shard}.jsonl").open("a", encoding="utf-8") as pending:
                                    pending.write(json.dumps(dict(id=qid, reason=reason), ensure_ascii=False) + "\n")
                            else:
                                relative = Path(record.get("image", "assets/" + qid + ".jpg"))
                                if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(re.escape(qid)+r'(?:-[a-f0-9]{12})?\.jpg',relative.name):
                                    raise ValueError("题图路径不合法")
                                asset = ROOT / relative
                                if ROOT.resolve() not in asset.resolve().parents:
                                    raise ValueError("题图路径越出题库目录")
                                box = record.get("bbox", [])
                                if len(box) != 4 or any(type(v) is not int for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                                    raise ValueError("题目缺少有效裁剪坐标")
                                stat = asset.stat()
                                fingerprint = [stat.st_size, stat.st_mtime_ns]
                                indexed_record = dict(record, image=str(asset.relative_to(ROOT)), fingerprint=fingerprint,
                                                      grade=record.get("dimensions", {}).get("source_grade"), paper_key=key,
                                                      unit="question", encoding="image_only", page_role="question_candidate")
                                old = db.execute("SELECT record FROM documents WHERE id=?", (qid,)).fetchone()
                                if old and json.loads(old[0]).get("fingerprint") == fingerprint:
                                    if json.loads(old[0]).get("bbox") != box:
                                        with Image.open(asset) as image:
                                            if image.size != (box[2]-box[0], box[3]-box[1]):
                                                raise ValueError("题图尺寸与裁剪坐标不一致")
                                    db.execute("UPDATE documents SET record=? WHERE id=?",
                                               (json.dumps(indexed_record,ensure_ascii=False),qid))
                                    progress["skipped_questions"] += 1
                                else:
                                    if not row:
                                        text = f"{record['subject']} {record['title']}"
                                        row = (json.dumps(dict(answer_ref=None, title=record["title"]), ensure_ascii=False),
                                               unit_vector(encoder.encode(texts=[text])[0]).astype("<f2").tobytes())
                                    db.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?)", (key, *row))
                                    with Image.open(asset) as image:
                                        if image.size != (box[2] - box[0], box[3] - box[1]):
                                            raise ValueError("题图尺寸与裁剪坐标不一致")
                                        picture = image.convert("RGB")
                                    vector = encoder.encode(images=[picture])[0]
                                    # ponytail: reuse existing crop boundaries; unresolved/cross-page records wait for review.
                                    unit = "question"
                                    store(db, indexed_record, vector,commit=False)
                                    progress["completed_questions"] += 1
                                    print("QUESTION_EMBED", qid, unit, flush=True)
                        except sqlite3.Error:
                            db.rollback()
                            raise  # Keep the last committed vector/offset pair after storage failure.
                        except Exception as error:
                            progress["failed_questions"] += 1
                            bad_id = record.get("id") if isinstance(record, dict) else None
                            with (DIRECTORY / f"errors-{shard}.jsonl").open("a", encoding="utf-8") as errors:
                                errors.write(json.dumps(dict(id=bad_id, offset=progress["offset"], error=repr(error)), ensure_ascii=False) + "\n")
                            print("QUESTION_ERROR", bad_id, repr(error), flush=True)
                        progress.update(offset=source.tell(), seen_records=progress["seen_records"] + 1, updated=time.time())
                        db.execute("INSERT OR REPLACE INTO settings VALUES ('crop_progress',?)", (json.dumps(progress),))
                        # ponytail: one atomic vector/offset checkpoint per 32 records or 5 seconds limits NFS commits.
                        if progress['seen_records']-checkpoint_seen>=32 or time.monotonic()-checkpoint_at>=5:
                            db.commit();atomic_json(progress_path,progress)
                            checkpoint_seen=progress['seen_records'];checkpoint_at=time.monotonic()
            db.commit()
            progress.update(state="finished" if not follow or finished else "waiting_for_crops", updated=time.time(),
                            indexed_questions=db.execute("SELECT count(*) FROM documents").fetchone()[0])
            atomic_json(progress_path, progress)
            if not follow or finished:
                return progress
            time.sleep(15)
    finally:
        db.rollback()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'merge', 'status'])
    parser.add_argument('--shard', type=int, choices=range(8))
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--max-pixels', type=int, default=MAX_PIXELS)
    args = parser.parse_args()
    if (args.command == 'build') != (args.shard is not None):
        parser.error('build 必须指定 --shard N；其他命令不接受 --shard')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.command == 'status':
        workers = []
        for shard in range(8):
            path = DIRECTORY / f'status-{shard}.json'
            workers.append(json.loads(path.read_text()) if path.exists() else dict(shard=shard, state='missing'))
        print(json.dumps(dict(workers=workers), ensure_ascii=False, indent=2))
        return
    import fcntl
    lock_name = 'publisher.lock' if args.command == 'merge' else f'process-{args.shard}.lock'
    with (DIRECTORY / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'merge':
            while True:
                states = [DIRECTORY / f'status-{i}.json' for i in range(8)]
                finished = all(p.exists() and json.loads(p.read_text())['state'] == 'finished' for p in states)
                try:
                    print('PUBLISHED', json.dumps(publish_index(), ensure_ascii=False), flush=True)
                except Exception as error:
                    if not args.follow:
                        raise
                    print('PUBLISH_RETRY', repr(error), flush=True)
                else:
                    if not args.follow or finished:
                        return
                time.sleep(60)
        else:
            db = connect(DIRECTORY / f'index-{args.shard}.sqlite3')
            try:
                check_encoder(db, args.max_pixels, initialize=True)
                progress = build_questions(db, Encoder(args.max_pixels), args.shard, args.follow)
                print('EMBED_FINISHED', json.dumps(progress, ensure_ascii=False), flush=True)
            finally:
                db.close()


if __name__ == '__main__':
    main()
