"""Shared source records, image references and question validation; no model loading."""
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def pin_image(image, qid, root=ROOT):
    """A stable hard-link name keeps working when the paper-folder name is moved."""
    if not re.fullmatch(r"[a-z0-9-]+", qid):
        raise ValueError("题目 ID 不合法")
    asset = root / "assets" / (qid + ".jpg")
    asset.parent.mkdir(parents=True, exist_ok=True)
    temporary = asset.with_suffix(".jpg.tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(image, temporary)
    temporary.replace(asset)
    return str(asset.relative_to(root))

def load_papers(config_path=None):
    import sys
    sys.path.insert(0, str(ROOT.parent / '北京题库'))
    from settings import load_config
    papers = []
    config = load_config(config_path)
    for subject, base in config['subjects'].items():
        with (base / "预览图片清单.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "已保存" or row['grade'] not in config['grades']:
                    continue
                if not row["id"].isdigit() or not 0 < int(row["pages"]) <= 10000:
                    raise ValueError("资料 ID 或页数无效")
                folder = (base / row["path"]).resolve()
                if base.resolve() not in folder.parents:
                    raise ValueError("资料路径越出原始题库")
                papers.append(dict(id=str(row["id"]), subject=subject, grade=row["grade"],
                                   title=row["title"], pages=int(row["pages"]), folder=str(folder)))
    return papers

def image_path(paper, page):
    return Path(paper["folder"]) / f"page-{page:0{len(str(paper['pages']))}d}.jpg"

def source_dimensions(paper):
    title = paper["title"]
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", title)
    regions = [r for r in ("东城", "西城", "海淀", "朝阳", "丰台", "石景山", "通州", "顺义",
                           "昌平", "大兴", "房山", "门头沟", "怀柔", "平谷", "密云", "延庆") if r in title]
    return {"subject": paper["subject"], "source_grade": paper["grade"], "source_years": years,
            "region": regions, "semester": "上学期" if "（上）" in title else "下学期" if "（下）" in title else None,
            "exam_type": [t for t in ("期中", "期末", "月考", "高考", "模拟", "汇编") if t in title]}


EXAM_NOTE = re.compile(r"^\s*(?:\d{1,3}\s*[.．、]\s*)?(?:本(?:试)?卷|试题答案一律|请将(?:试题)?答案|考生(?:务必|必须|须)|答题(?:前|时|后)|考试结束|注意事项)")
ANSWER_CUE = re.compile(r"【(?:答案|解析|详解|分析|解答|点睛|(?:参考)?范文)】|【作者\s*ChatGPT】|^[^\S\r\n]*\d{1,3}[^\S\r\n]*[.．、][^\S\r\n]*(?:答案\s*[:：]|[A-G][^\S\r\n]*(?:。|$)|考查[^\n]{0,30}[。．.]\s*句意[:：])|^[^\S\r\n]*(?:解|解析|解答|参考范文)[^\S\r\n]*[:：]|^[^\S\r\n]*(?:(?:参考)?范文|(?:One\s+)?possible\s+version)\s*[:：]?\s*$",re.I|re.M)
SOLUTION_CUE = re.compile(ANSWER_CUE.pattern+r'|^[^\S\r\n]*提示[^\S\r\n]*[:：]',re.I|re.M)


def is_exam_instruction(text):
    return bool(EXAM_NOTE.match(text))


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
