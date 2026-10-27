"""Encode existing question crops; --pages explicitly selects the legacy page index."""
import argparse
import contextlib
import html
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import ROOT, atomic_json, image_path, load_papers, pin_image
from pdf_native import is_exam_instruction,ANSWER_CUE

MODEL = ROOT / "models/Qwen3-VL-Embedding-2B"
DIRECTORY = ROOT / "embedding"
INSTRUCTION = "Represent the user's input."
TOPICS = {
    "数学": ["三角函数", "空间向量", "概率", "导数", "数列", "集合", "圆锥曲线", "统计"],
    "英语": ["完形", "阅读理解", "语法填空", "写作", "七选五", "非谓语", "定语从句", "阅读表达"],
}
# Soft preferences from free native PDF text; unmatched requirements still retain vector candidates.
QUESTION_PREFERENCES=[
    ('线面角',r'线面角|直线.{0,30}平面.{0,15}夹角',r'直线.{0,80}平面.{0,30}(?:夹角|所成.{0,5}角)',.14),
    ('几何体',r'棱锥|棱柱|正方体|长方体',r'棱锥|棱柱|正方体|长方体',.08),
    ('极值',r'极大值|极小值|极值',r'极大值|极小值|极值',.12),
    ('列出分布列',r'(?:列出|求).{0,30}分布列',r'(?:求|列出|写出).{0,45}分布列',.16),
    ('数学期望',r'期望',r'期望|\bE[^A-Za-z0-9]{0,3}[XYZξ]',.08),
    ('动词形式填空',r'(?:填入|填空).{0,25}(?:动词|正确形式)|动词.{0,15}(?:填空|正确形式)',r'语法填空|(?:所给|括号[内中]|动词).{0,15}形式|correct form',.12),
    ('邀请',r'邀请',r'邀请|\binvit(?:e|ation|ing)\b',.10),
    ('科技主题',r'科技|人工智能',r'\bAI\b|artificial intelligence|\btechnolog(?:y|ies)\b',.08),
    ('等差数列',r'等差',r'等差',.06),
    ('椭圆',r'椭圆',r'椭圆',.08),('双曲线',r'双曲线',r'双曲线',.08),('抛物线',r'抛物线',r'抛物线',.08),
    ('条件概率',r'条件概率',r'条件概率|在.{0,20}条件下.{0,20}概率',.08),
]
NATURAL_QUERIES = [
    ("我想给高一学生找几道三角函数练习，最好带正弦函数图像，考查周期、平移或单调性，难度适中。", "三角函数"),
    ("帮我找一些立体几何题，图里有空间几何体，需要利用向量计算直线与平面的夹角，适合高二复习。", "空间向量"),
    ("给我几道导数应用题，要让学生判断函数在哪些区间递增递减，再求极大值或极小值。", "导数"),
    ("我想出一份集合运算的小练习，包含交集、并集、补集，最好是选择题。", "集合"),
    ("找一道高中英语应用文写作题，要求给外国朋友写信或邮件，可以是邀请对方参加活动，保留原题截图。", "写作"),
    ("给高二学生找一篇英语阅读，主题是老年人使用手机和社交媒体、屏幕时间对生活的影响，带阅读理解题。", "阅读理解"),
    ("找英语关系从句的练习，重点是 which、that、whose 的用法，可以包含句子填空。", "定语从句"),
    ("我想练习英语阅读里的完形填空，给出一篇有若干空格的短文，每个空格有 A、B、C、D 四个选项。", "完形"),
]


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


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("向量必须是一维有限数值")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("不能索引零向量")
    return vector / norm


def store(db, record, vector,commit=True):
    blob = unit_vector(vector).astype("<f2").tobytes()
    with db if commit else contextlib.nullcontext():
        db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?)",
                   (record["id"], json.dumps(record, ensure_ascii=False), blob))


def sparse_fragment(path):
    """Keep near-blank continuation pages out of standalone question results."""
    import cv2
    image = cv2.imread(str(path), 0)
    if image is None:
        return True
    height, width = image.shape
    body = image[int(.04 * height):int(.93 * height), int(.06 * width):int(.95 * width)] < 160
    rows = np.flatnonzero(body.any(1))
    # ponytail: publisher-specific geometry; retain the source image for later cross-page assembly.
    return not rows.size or (body.mean() < .005 and rows[-1] + .04 * height < .25 * height)


def prepare_search(db,device=None):
    """Load one immutable search snapshot once for a batch of queries."""
    rows = db.execute("SELECT record,vector FROM documents ORDER BY id").fetchall()
    paper_rows = db.execute("SELECT id,record,vector FROM papers").fetchall()
    papers = {row[0]: (json.loads(row[1]), unit_vector(np.frombuffer(row[2], dtype="<f2"))) for row in paper_rows}
    question_index=db.execute("SELECT value FROM settings WHERE key='unit'").fetchone()==('question',)
    # ponytail: exact cosine search fits the current 157k pages; use ANN only if measured latency requires it.
    matrix = np.stack([np.frombuffer(row[1], dtype="<f2") for row in rows]) if rows else np.empty((0,0),dtype='<f2')
    if device:
        import torch
        matrix=torch.from_numpy(matrix).to(device=device).float()
        norms=torch.linalg.vector_norm(matrix,dim=1)
    else:
        matrix=matrix.astype(np.float32);norms=np.linalg.norm(matrix,axis=1)
    return ([json.loads(row[0]) for row in rows],matrix,norms,papers,question_index)


def rank(db, vector, limit=5, query_text="",corpus=None):
    query = unit_vector(vector)
    if limit<=0:return []
    records,matrix,norms,papers,question_index=corpus if corpus is not None else prepare_search(db)
    if not records:return []
    if matrix.shape[1] != query.size:
        raise ValueError("查询向量和索引维度不一致")
    if isinstance(matrix,np.ndarray):
        image_scores = (matrix @ query) / norms
    else:
        import torch
        image_scores=((matrix @ torch.as_tensor(query,device=matrix.device))/norms).cpu().numpy()
    paper_scores={key:float(value[1] @ query) for key,value in papers.items()}
    metadata_scores = np.array([paper_scores.get(r.get('paper_key'),0) for r in records])
    scores = 0.65 * image_scores + 0.35 * metadata_scores if papers else image_scores
    terms = [(term, 0.15) for topics in TOPICS.values() for term in topics]
    terms += [(term, 0.04) for term in ("高一", "高二", "高三")]
    terms += [(term, 0.02) for term in ("数学", "英语")]
    normalized_query = query_text.replace("关系从句", "定语从句")
    terms=[(term,weight) for term,weight in terms if term in normalized_query]
    matches = [[term for term, _ in terms if term in normalized_query and term in r.get("title", "")]
               for r in records]
    for i in range(len(records)):
        scores[i] += sum(weight for term, weight in terms if term in matches[i])
    order = np.argsort(-scores, kind="stable")
    preferred=[(name,re.compile(document,re.I),weight) for name,query_pattern,document,weight in QUESTION_PREFERENCES
               if re.search(query_pattern,re.sub(r'\s+','',query_text),re.I)]
    question_matches={}
    if preferred:
        # ponytail: rerank 1000 vector candidates; widen only if measured recall requires it.
        for i in order[:1000]:
            original=' '.join(records[i].get('text','').split())
            text=original.replace(' ','')
            matched=[(name,weight) for name,pattern,weight in preferred if pattern.search(text) or pattern.search(original)]
            scores[i]+=min(.28,sum(weight for _,weight in matched))
            question_matches[i]=[name for name,_ in matched]
        order=np.argsort(-scores,kind='stable')
    if question_index:
        # Validate ranked candidates until enough pass; repeating text checks for the entire corpus adds no safety.
        selected=[];seen_text=set()
        for i in order:
            if question_skip_reason(records[i],papers.get(records[i].get('paper_key'),({},))[0]) is not None:continue
            text=records[i].get('text','')
            if not (records[i].get('subject')=='英语' and records[i].get('coordinate_space')=='composite_image'):
                text=re.sub(r'(?m)^[^\S\r\n]*\d{1,3}[^\S\r\n]*[.．、]','',text,count=1)
            signature=re.sub(r'\s+','',text) or records[i]['id']
            if signature in seen_text:continue
            selected.append(i);seen_text.add(signature)
            if len(selected)>=limit:break
        order=selected
    if query_text:
        order = [i for i in order[:max(20, limit * 4)]
                 if records[i].get("image") and (records[i].get("unit") != "page" or
                                                 not sparse_fragment(ROOT / records[i]["image"]))]
    order = order[:limit]
    # ponytail: a relative cutoff calibrated on this publisher's pilot; expand the labeled query set before changing it.
    if query_text and len(order):
        order = [i for i in order if scores[i] >= 0.85 * scores[order[0]]]
    found = []
    for i in order:
        record = dict(records[i], score=round(float(scores[i]), 6), image_score=round(float(image_scores[i]), 6),
                      matched_source_terms=matches[i],matched_question_terms=question_matches.get(i,[]))
        if record.get("paper_key") in papers:
            record.update(metadata_score=round(float(metadata_scores[i]), 6),
                          answer_ref=papers[record["paper_key"]][0].get("answer_ref"))
        found.append(record)
    return found


def answer_boundary(paper):
    """Find the publisher's reference-answer heading, without text recognition."""
    import cv2
    cv2.setNumThreads(2)
    reference = cv2.imread(str(ROOT / "assets/math-195356-p003-page.jpg"), 0)
    if reference is None:
        raise ValueError("答案标题参考图缺失，不能静默将答案纳入题目索引")
    factor = 900 / reference.shape[1]
    x1, y1, x2, y2 = [round(v * factor) for v in (680, 132, 874, 190)]
    reference = cv2.resize(reference, (900, round(reference.shape[0] * factor)))[y1:y2, x1:x2]
    templates = [cv2.resize(reference, None, fx=scale, fy=scale) for scale in (0.9, 1, 1.1)]
    # ponytail: this matches the current publisher's heading/layout; other publishers need their own templates or a classifier.
    for page in range(1, paper["pages"] + 1):
        image = cv2.imread(str(image_path(paper, page)), 0)
        if image is None:
            raise ValueError("页面无法读取: " + str(image_path(paper, page)))
        image = cv2.resize(image, (900, round(image.shape[0] * 900 / image.shape[1])))
        region = image[:round(image.shape[0] * 0.4), 180:720]
        score = max(cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))[1]
                    for template in templates)
        if score >= 0.70:
            return dict(page=page, score=round(score, 4), method="publisher_answer_heading_v1")
    return None


def pin_paper_answers(paper, boundary):
    if not boundary:
        return None
    subject = "math" if paper["subject"] == "数学" else "english"
    images = []
    for page in range(boundary["page"], paper["pages"] + 1):
        qid = f"{subject}-{paper['id']}-p{page:03d}-page"
        asset = ROOT / "assets" / (qid + ".jpg")
        if not asset.exists():
            pin_image(image_path(paper, page), qid)
        images.append(dict(page=page, image=str(asset.relative_to(ROOT))))
    return dict(paper_id=paper["id"], match_level="paper", question_number_status="not_aligned",
                boundary=boundary, pages=images)


def prepare_paper(db, paper, vector):
    boundary = answer_boundary(paper)
    key = paper["subject"] + ":" + paper["id"]
    record = dict(paper_id=paper["id"], subject=paper["subject"], title=paper["title"],
                  answer_ref=pin_paper_answers(paper, boundary), boundary_status="detected" if boundary else "not_detected")
    old = db.execute("SELECT record FROM papers WHERE id=?", (key,)).fetchone()
    if old and record["answer_ref"]:
        pairs = (json.loads(old[0]).get("answer_ref") or {}).get("verified_question_pairs")
        if pairs:
            record["answer_ref"]["verified_question_pairs"] = pairs
    prefix = ("math" if paper["subject"] == "数学" else "english") + "-" + paper["id"] + "-p"
    removed = 0
    with db:
        db.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?)",
                   (key, json.dumps(record, ensure_ascii=False), unit_vector(vector).astype("<f2").tobytes()))
        for qid, content in db.execute("SELECT id,record FROM documents WHERE id>=? AND id<?", (prefix, prefix + "\uffff")).fetchall():
            r = json.loads(content)
            if not boundary or r["page"] >= boundary["page"]:
                db.execute("DELETE FROM documents WHERE id=?", (qid,))
                removed += 1
            else:
                r.update(paper_key=key, page_role="question_candidate")
                db.execute("UPDATE documents SET record=? WHERE id=?", (json.dumps(r, ensure_ascii=False), qid))
    return key, boundary, removed


def refine(db, encoder):
    import shutil
    for suffix in ("json", "html"):
        source = DIRECTORY / ("natural." + suffix)
        if source.exists() and not (DIRECTORY / ("natural-unfiltered." + suffix)).exists():
            shutil.copy2(source, DIRECTORY / ("natural-unfiltered." + suffix))
    papers = selected_papers()
    texts = [f"{paper['subject']} {paper['grade']} {paper['title']}" for paper, _, _ in papers]
    vectors = encoder.encode(texts=texts)
    summary = []
    for (paper, _, _), vector in zip(papers, vectors):
        _, boundary, removed = prepare_paper(db, paper, vector)
        summary.append(dict(paper_id=paper["id"], subject=paper["subject"], boundary=boundary, removed_pages=removed))
    atomic_json(DIRECTORY / "answer-boundaries.json", summary)
    print("REFINED", json.dumps(summary, ensure_ascii=False), flush=True)


class Encoder:
    def __init__(self, max_pixels):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoProcessor, Qwen3VLModel
        torch.set_num_threads(2)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Keep room for the already running OCR worker on the same GPU.
            torch.cuda.set_per_process_memory_fraction(0.25)
        started = time.perf_counter()
        self.model, info = Qwen3VLModel.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa", output_loading_info=True)
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError("Embedding 权重不匹配: " + str(info))
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, padding_side="right")
        self.processor.image_processor.size = {"shortest_edge": 4096, "longest_edge": max_pixels}
        self.load_seconds = round(time.perf_counter() - started, 3)
        print("EMBED_MODEL_READY", self.device, self.load_seconds, flush=True)

    def encode(self, texts=None, images=None):
        texts = texts if texts is not None else [None] * len(images)
        images = images if images is not None else [None] * len(texts)
        if not texts or len(texts) != len(images):
            raise ValueError("编码输入数量不一致或为空")
        conversations, pictures = [], []
        for text, picture in zip(texts, images):
            content = []
            if picture is not None:
                content.append({"type": "image"})
                pictures.append(picture)
            if text:
                content.append({"type": "text", "text": text})
            if not content:
                raise ValueError("编码内容为空")
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": content}])
        rendered = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=rendered, images=pictures or None, padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 8192:
            raise ValueError("输入过长，未截断图片或文字")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"]
            last = mask.shape[1] - mask.flip(1).argmax(1) - 1
            vectors = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), last].float()
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors.cpu().numpy()


def selected_papers(all_pages=False, shard=None):
    papers = (json.loads((ROOT / "run" / f"shard-{shard}.json").read_text())
              if all_pages and shard is not None else load_papers())
    if all_pages:
        return [(paper, range(1, paper["pages"] + 1), None) for paper in papers]
    selected, used = [], set()
    for subject, topics in TOPICS.items():
        for i, topic in enumerate(topics):
            candidates = [p for p in papers if p["subject"] == subject and topic in p["title"]
                          and p["pages"] >= 4 and (subject, p["id"]) not in used]
            if not candidates:
                raise ValueError("没有找到样本主题: " + topic)
            grade = ["高一", "高二", "高三"][i % 3]
            paper = next((p for p in candidates if p["grade"] == grade), candidates[0])
            used.add((subject, paper["id"]))
            selected.append((paper, range(1, 5), topic))
    return selected


def build(db, encoder, all_pages=False, shard=None):
    from PIL import Image
    papers = selected_papers(all_pages, shard)
    progress_path = DIRECTORY / (f"status-{shard}.json" if shard is not None else "status.json")
    errors_path = DIRECTORY / (f"errors-{shard}.jsonl" if shard is not None else "errors.jsonl")
    progress = dict(started=time.time(), pid=os.getpid(), shard=shard, state="running", total_pages=sum(len(pages) for _, pages, _ in papers),
                    completed_pages=0, skipped_pages=0, failed_pages=0, excluded_answer_pages=0, review_pages=0,
                    encoding_seconds=0, page_seconds=[])
    for paper, pages, topic in papers:
        subject = "math" if paper["subject"] == "数学" else "english"
        progress.update(paper_id=paper["id"], title=paper["title"])
        atomic_json(progress_path, dict(progress, updated=time.time()))
        try:
            key = paper["subject"] + ":" + paper["id"]
            cached = db.execute("SELECT record FROM papers WHERE id=?", (key,)).fetchone()
            if cached:
                boundary = (json.loads(cached[0]).get("answer_ref") or {}).get("boundary")
            else:
                text = f"{paper['subject']} {paper['grade']} {paper['title']}"
                key, boundary, _ = prepare_paper(db, paper, encoder.encode(texts=[text])[0])
        except Exception as error:
            progress["failed_pages"] += len(pages)
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(paper_id=paper["id"], error=repr(error)), ensure_ascii=False) + "\n")
            print("PAPER_REVIEW", paper["id"], repr(error), flush=True)
            atomic_json(progress_path, dict(progress, updated=time.time()))
            continue
        if not boundary:
            # Unrecognized layouts wait for review instead of leaking unknown answer pages into the question index.
            progress["review_pages"] += len(pages)
            print("PAPER_REVIEW", paper["id"], "answer_boundary_not_detected", flush=True)
            atomic_json(progress_path, dict(progress, updated=time.time()))
            continue
        for page in pages:
            progress["page"] = page
            if page >= boundary["page"]:
                progress["excluded_answer_pages"] += 1
                continue
            qid = f"{subject}-{paper['id']}-p{page:03d}-page"
            asset = ROOT / "assets" / (qid + ".jpg")
            started = time.perf_counter()
            try:
                if not asset.exists():
                    pin_image(image_path(paper, page), qid)
                stat = asset.stat()
                fingerprint = [stat.st_size, stat.st_mtime_ns]
                old = db.execute("SELECT record FROM documents WHERE id=?", (qid,)).fetchone()
                if old and json.loads(old[0])["fingerprint"] == fingerprint:
                    progress["skipped_pages"] += 1
                    continue
                with Image.open(asset) as original:
                    picture = original.convert("RGB")
                vector = encoder.encode(images=[picture])[0]
                record = dict(id=qid, image=str(asset.relative_to(ROOT)), subject=paper["subject"],
                              grade=paper["grade"], paper_id=paper["id"], title=paper["title"], page=page,
                              source_image=str(image_path(paper, page)), fingerprint=fingerprint,
                              sample_topic=topic, unit="page", encoding="image_only", paper_key=key,
                              page_role="question_candidate")
                store(db, record, vector)
                seconds = round(time.perf_counter() - started, 3)
                progress["completed_pages"] += 1
                progress["encoding_seconds"] = round(progress["encoding_seconds"] + seconds, 3)
                progress["page_seconds"].append(seconds)
                progress["page_seconds"] = progress["page_seconds"][-100:]
                print("EMBED_PAGE", qid, seconds, flush=True)
            except Exception as error:
                progress["failed_pages"] += 1
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(id=qid, error=repr(error)), ensure_ascii=False) + "\n")
                print("EMBED_ERROR", qid, repr(error), flush=True)
            finally:
                atomic_json(progress_path, dict(progress, updated=time.time()))
    progress.update(state="finished", elapsed_seconds=round(time.time() - progress["started"], 3),
                    indexed_pages=db.execute("SELECT count(*) FROM documents").fetchone()[0])
    atomic_json(progress_path, dict(progress, updated=time.time()))
    return progress


def question_skip_reason(record, paper_record=None):
    if record.get("crop_method") == "native_pdf_coordinates" and record.get("version") != "native-crops-v2":
        return "review"
    if record.get("role") != "question":
        return "answer_or_other"
    boundary = ((paper_record or {}).get("answer_ref") or {}).get("boundary")
    if record.get("crop_method") == "native_pdf_coordinates":
        boundary = record.get("native_answer_boundary") or boundary
    if boundary and record["page"] >= boundary["page"]:
        boxes = record.get("source_boxes", [])
        before_heading = (record.get("crop_method") == "native_pdf_coordinates" and record["page"] == boundary["page"]
                          and boxes and all(box[3] < boundary.get("source_y", 0) for box in boxes))
        if not before_heading:
            return "answer_or_other"
    text = record.get("text", "")
    if record.get('subject')=='英语':
        if '完形' in record.get('title','') and re.match(r'^\s*[（(]\s*\d{1,3}\s*[）)]\s*(?:_+|[a-z]|[，,;；])',text):return 'cross_page'
        if re.search(r'内容(?:包括|包含)\s*[:：]\s*$',text):return 'review'
        if any(t in record.get('title','') for t in ('写作','书面表达')) and len(text)<100 and not re.search(r'假[设如定]|你是|你校|请(?:你|根据|以)|写[一篇封]|\bWrite\b|\bSuppose\b',text,re.I):
            return 'review'
    if is_exam_instruction(text):
        return "review"
    if ANSWER_CUE.search(text) or re.search(r"考查.{0,8}词义辨析|故选\s*[A-D]|正确答案[：:]", text,re.I):
        return "answer_or_other"
    if record.get("status") == "cross_page":
        return "cross_page"
    if record.get("status") != "complete":
        return "review"
    if not str(record.get("number", "")).isdigit() and record.get('number_source')!='native_prompt':
        return "review"  # Multi-question groups need to be split before entering this index.
    if record.get("subject") == "英语" and re.match(r"\s*\d{1,3}\s*[.．、]\s*[A-D]\s*[.．、]", text):
        return "cross_page"
    if record.get("subject") == "数学":
        starts = re.findall(r"(?m)^[^\S\r\n]*(\d{1,3})[^\S\r\n]*[.．、](?!\d)", text)
        if (not starts and record.get('number_source')!='native_prompt') or len(set(starts)) > 1:
            return "review"
        if starts and record.get('number_source')!='native_prompt' and str(record.get('number')) not in starts:
            return 'review'
    else:
        kind = str(record.get("dimensions", {}).get("question_type", ""))
        requires_material = any(term in record.get("title", "") + kind for term in ("阅读", "完形", "七选五", "reading", "cloze"))
        requires_material |= bool(re.search(r"\b(?:passage|paragraph|writer|author|according to|main idea|best title|(?:the|this)\s+text)\b", text, re.I))
        native_context=record.get('native_context_state') in ('same_page_material','embedded_material')
        if requires_material and not native_context and len(re.findall(r"\b[A-Za-z]+\b", text)) < 100:
            return "cross_page"
        if ("完形" in kind or "完形" in record.get("title", "")) and not record.get('native_task') and not all(
                re.search(r"\b" + choice + r"[.．、]", text) for choice in "ABCD"):
            return "cross_page"
        if len(re.findall(r"\bPassage\s*\d", text, re.I)) > 1:
            return "review"
    return None


def build_questions(db, encoder, shard, follow=False, source_directory=None):
    """Consume the existing crop journal; never OCR or encode a source page here."""
    from PIL import Image
    source_directory = source_directory or ROOT / "run"
    journal = source_directory / f"results-{shard}.jsonl"
    progress_path = DIRECTORY / f"status-{shard}.json"
    saved = db.execute("SELECT value FROM settings WHERE key='crop_progress'").fetchone()
    progress = json.loads(saved[0]) if saved and json.loads(saved[0]).get("version") == "questions-v1" else dict(
        version="questions-v1", offset=0, seen_records=0, completed_questions=0,
        skipped_questions=0, excluded_answer_records=0, deferred_cross_page=0, deferred_review=0, failed_questions=0)
    progress.update(started=time.time(), pid=os.getpid(), shard=shard, state="running",checkpoint_policy='32_records_or_5_seconds')
    metadata_path = ROOT / "embedding" / f"index-{shard}.sqlite3"
    metadata = (sqlite3.connect(metadata_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
                if metadata_path.exists() else None)
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
                            # ponytail: reuse prior paper vectors when present; fresh installs cache titles in this shard.
                            row = (metadata or db).execute("SELECT record,vector FROM papers WHERE id=?", (key,)).fetchone()
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
        if metadata is not None:
            metadata.close()


def demo(db, encoder, queries=None, name="demo", vectors=None,device=None):
    queries = queries or [
        ("三角函数的图像、周期和单调性", "三角函数"),
        ("用空间向量求直线与平面的夹角", "空间向量"),
        ("随机事件的概率计算", "概率"),
        ("利用导数求函数的极值和单调区间", "导数"),
        ("等差数列的通项公式和前n项和", "数列"),
        ("集合的交集、并集和补集", "集合"),
        ("椭圆双曲线与直线的位置关系", "圆锥曲线"),
        ("样本数据与统计分析", "统计"),
        ("英语完形填空，根据上下文选择单词", "完形"),
        ("英语阅读理解，文章和四个选项", "阅读理解"),
        ("英语语法填空，在空白处填入单词的正确形式", "语法填空"),
        ("英语书信写作，写一封邀请信", "写作"),
        ("英语七选五，从七个句子中选择五个填入文章", "七选五"),
        ("英语非谓语动词练习，不定式和动名词", "非谓语"),
        ("英语定语从句，关系代词which和that", "定语从句"),
        ("英语阅读表达，阅读文章后用英语回答问题", "阅读表达"),
        ("三角函数的图像 适中难度 不存在的关键词", "三角函数"),
    ]
    reused_vectors = vectors is not None
    started = time.perf_counter()
    if vectors is None:
        vectors = encoder.encode(texts=[q for q, _ in queries])
    if vectors.ndim!=2 or len(vectors)!=len(queries):raise ValueError('查询向量数量与固定查询清单不一致')
    query_seconds = time.perf_counter() - started
    np.save(DIRECTORY / (name + "-queries.npy"), vectors, allow_pickle=False)
    results = []
    started=time.perf_counter()
    corpus=prepare_search(db,device=device or getattr(encoder,'device',None))
    print('QUERY_SNAPSHOT_READY',len(corpus[0]),round(time.perf_counter()-started,3),flush=True)
    for (query, topic), vector in zip(queries, vectors):
        started=time.perf_counter()
        found = rank(db, vector, 5, query_text=query,corpus=corpus)
        print('QUERY_TOP5',query,round(time.perf_counter()-started,3),[r['id'] for r in found],flush=True)
        results.append(dict(query=query, expected_source_topic=topic, results=found,
                            source_topic_in_top3=any(r.get("sample_topic") == topic for r in found[:3]),
                            source_topic_in_top5=any(r.get("sample_topic") == topic for r in found)))
    unit = db.execute("SELECT value FROM settings WHERE key='unit'").fetchone()
    report = dict(model=MODEL.name, dimensions=int(vectors.shape[1]), encoding="image_only", unit=unit[0] if unit else "page",
                  query_batch_seconds=None if reused_vectors else round(query_seconds, 3), query_vectors_reused=reused_vectors,
                  source_topic_in_top3=sum(r["source_topic_in_top3"] for r in results), queries=len(results),
                  source_topic_in_top5=sum(r["source_topic_in_top5"] for r in results),
                  note=("来源主题检查仅用于试运行，不代表逐题准确率；按出版社答案标题分界排除解析页；"
                        "评分为图片向量65%与现有目录文字向量35%，并对命中的已知章节/学科/年级加分；"
                        "仅返回达到本次最高分85%的候选，最多5条。答案仅关联到同一套资料，尚未按题号对齐。"
                        if db.execute("SELECT count(*) FROM papers").fetchone()[0] else
                        "来源主题检查仅用于试运行，不代表逐题准确率；元数据未输入模型或参与相似度评分。"), results=results)
    report['source_topic_checks_available']=any('sample_topic' in r for item in results for r in item['results'])
    if not report['source_topic_checks_available']:
        report['source_topic_in_top3']=report['source_topic_in_top5']=None
        for item in results:item['source_topic_in_top3']=item['source_topic_in_top5']=None
    report['note']+=' 原生题干中命中的明确题型和要求用于候选软加分，未知词不排除候选；逐字相同题干的重复汇编只返回一份。最终质量以逐图人工核验为准。'
    atomic_json(DIRECTORY / (name + ".json"), report)
    review_path = DIRECTORY / "manual-review.json"
    reviews = ({(q["query"], r["id"]): r["assessment"] for q in json.loads(review_path.read_text())["queries"]
                for r in q["results"]} if review_path.exists() else {})
    body = ['<!doctype html><meta charset="utf-8"><title>图片向量检索样本</title>',
            '<style>body{font:16px sans-serif;margin:24px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}figure{margin:0}img{width:100%}h2{margin-top:40px}</style>',
            '<h1>原始截图 → 2048 维向量 → 中文检索</h1>', '<p>' + html.escape(report["note"]) + '</p>']
    for item in results:
        body.append('<h2>' + html.escape(item["query"]) + '</h2><section>')
        for result in item["results"]:
            url = "../" + result["image"]
            body.append('<figure><figcaption>' + html.escape(f"{result['score']:.3f} · {result['title']} · 第{result['page']}页") +
                        '</figcaption><a href="' + html.escape(url, quote=True) + '"><img loading="lazy" src="' +
                        html.escape(url, quote=True) + '" alt="' + html.escape(result["title"], quote=True) + '"></a></figure>')
            assessment = reviews.get((item["query"], result["id"]))
            if assessment:
                body[-1] = body[-1].replace('</figcaption>', '</figcaption><p>目视核验：' + html.escape(assessment) + '</p>')
            ref = result.get("answer_ref")
            if ref and ref.get("pages"):
                answer = ref["pages"][0]
                body[-1] = body[-1].replace('</figure>', '<p><a href="../' + html.escape(answer["image"], quote=True) +
                                           '">同套答案：第' + str(answer["page"]) + '页起（题号待对齐）</a></p></figure>')
        body.append('</section>')
    (DIRECTORY / (name + ".html")).write_text("\n".join(body), encoding="utf-8")
    print("EMBED_DEMO", json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False), flush=True)


def main():
    global DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sample", "build", "search", "natural", "refine", "status", "merge"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(8))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--pages", action="store_true", help="显式使用旧的整页索引；默认逐题截图索引")
    parser.add_argument("--native-crops", action="store_true", help="只使用 PDF 原生坐标直接裁剪的单题图，不使用 OCR 裁剪")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
    args = parser.parse_args()
    if args.pages and args.native_crops:
        parser.error("整页和 PDF 原生单题索引不能混用")
    if not args.pages:
        if args.command in ("search", "natural", "status") and (ROOT / "embedding-native-questions/index.sqlite3").exists():
            args.native_crops = True
        DIRECTORY = ROOT / ("embedding-native-questions" if args.native_crops else "embedding-questions")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not args.pages and args.command in ("sample", "refine"):
        parser.error("逐题索引直接使用已有裁剪；旧的整页试验需显式添加 --pages")
    if args.command == "build" and not args.pages and (not args.all or args.shard is None):
        parser.error("逐题批处理请使用 build --all --shard N --follow")
    if args.command == "merge":
        import fcntl
        with (DIRECTORY / "publisher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                states = [DIRECTORY / f"status-{i}.json" for i in range(8)]
                finished = all(p.exists() and json.loads(p.read_text())["state"] == "finished" for p in states)
                try:
                    print("PUBLISHED", json.dumps(publish_index(DIRECTORY), ensure_ascii=False), flush=True)
                except Exception as error:
                    if not args.follow:
                        raise
                    print("PUBLISH_RETRY", repr(error), flush=True)
                else:
                    if not args.follow or finished:
                        break
                time.sleep(60)
        return
    if args.command == "status":
        workers = []
        for shard in range(8):
            path = DIRECTORY / f"status-{shard}.json"
            state = json.loads(path.read_text()) if path.exists() else dict(shard=shard, state="loading")
            state.pop("page_seconds", None)
            state.setdefault("updated", path.stat().st_mtime if path.exists() else None)
            workers.append(state)
        keys = (("total_pages", "completed_pages", "skipped_pages", "failed_pages", "excluded_answer_pages", "review_pages") if args.pages else
                ("seen_records", "completed_questions", "skipped_questions", "excluded_answer_records",
                 "deferred_cross_page", "deferred_review", "failed_questions"))
        totals = {key: sum(w.get(key, 0) for w in workers) for key in keys}
        if (DIRECTORY / "published.json").exists():
            totals.update(json.loads((DIRECTORY / "published.json").read_text()))
        print(json.dumps(dict(state="finished" if all(w["state"] == "finished" for w in workers) else "running",
                              **totals, workers=workers), ensure_ascii=False, indent=2))
        return
    if (args.command in ("sample", "refine") or (args.command == "build" and args.shard is None)) and (DIRECTORY / "index-0.sqlite3").exists():
        parser.error("已启用分片存储，发布后的查询快照只读；批量构建请使用 build --all --shard N")
    if args.shard is not None and (args.command != "build" or not args.all):
        parser.error("shard 仅用于 build --all")
    if not 4096 <= args.max_pixels <= 1800 * 32 * 32:
        parser.error("max-pixels 必须在 4096 到 1843200 之间")
    if args.command == "search" and not args.query.strip():
        print("[]")
        return
    if len(args.query) > 4096:
        parser.error("查询过长")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (DIRECTORY / (f"process-{args.shard}.lock" if args.shard is not None else "process.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        readonly = args.command in ("search", "natural")
        path = DIRECTORY / (f"index-{args.shard}.sqlite3" if args.shard is not None else "index.sqlite3")
        db = (sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
              if readonly else connect(path))
        manifest = json.loads((MODEL / "download-manifest.json").read_text())
        checksum = next(f["Sha256"] for f in manifest if f["Path"] == "model.safetensors")
        settings = json.dumps(dict(model=MODEL.name, sha256=checksum, max_pixels=args.max_pixels,
                                   instruction=INSTRUCTION, pooling="last_nonpadding", dtype="float16"), sort_keys=True)
        old = db.execute("SELECT value FROM settings WHERE key='encoder'").fetchone()
        if old and old[0] != settings:
            raise ValueError("编码设置不同，请使用独立索引，不能混用向量")
        if not readonly:
            with db:
                db.execute("INSERT OR IGNORE INTO settings VALUES ('encoder',?)", (settings,))
                db.execute("INSERT OR IGNORE INTO settings VALUES ('unit',?)", ("page" if args.pages else "question",))
        encoder = Encoder(args.max_pixels)
        if args.command in ("sample", "build"):
            progress = (build(db, encoder, all_pages=args.all, shard=args.shard) if args.pages
                        else build_questions(db, encoder, args.shard, args.follow,
                                             ROOT / "pdf-fast/crops-run" if args.native_crops else None))
            print("EMBED_FINISHED", json.dumps(progress, ensure_ascii=False), flush=True)
            if args.command == "sample" and progress["indexed_pages"]:
                demo(db, encoder)
        elif args.command == "refine":
            refine(db, encoder)
            demo(db, encoder, NATURAL_QUERIES, "natural")
        elif args.command == "natural":
            demo(db, encoder, NATURAL_QUERIES, "natural")
        else:
            found = rank(db, encoder.encode(texts=[args.query])[0], args.limit, query_text=args.query,
                         corpus=prepare_search(db,device=encoder.device))
            print(json.dumps(found, ensure_ascii=False, indent=2))
        db.close()


if __name__ == "__main__":
    main()
