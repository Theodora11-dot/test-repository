"""Audit existing authorized PDF previews for usable native coordinates; never OCR."""
import argparse
import collections
import concurrent.futures
import json
import math
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

from pipeline import ROOT, atomic_json, load_papers

DIRECTORY = ROOT / "pdf-fast"
NS = {"x": "http://www.w3.org/1999/xhtml"}
STOP = threading.Event()
AUDIT_VERSION = "native-v3"
EXAM_NOTE = re.compile(r"^\s*(?:\d{1,3}\s*[.．、]\s*)?(?:本(?:试)?卷|试题答案一律|请将(?:试题)?答案|考生(?:务必|必须|须)|答题(?:前|时|后)|考试结束|注意事项)")
ANSWER_CUE = re.compile(r"【(?:答案|解析|详解|分析|解答|点睛|(?:参考)?范文)】|【作者\s*ChatGPT】|^[^\S\r\n]*\d{1,3}[^\S\r\n]*[.．、][^\S\r\n]*(?:答案\s*[:：]|[A-G][^\S\r\n]*(?:。|$)|考查[^\n]{0,30}[。．.]\s*句意[:：])|^[^\S\r\n]*(?:解|解析|解答|参考范文)[^\S\r\n]*[:：]|^[^\S\r\n]*(?:(?:参考)?范文|(?:One\s+)?possible\s+version)\s*[:：]?\s*$",re.I|re.M)
SOLUTION_CUE = re.compile(ANSWER_CUE.pattern+r'|^[^\S\r\n]*提示[^\S\r\n]*[:：]',re.I|re.M)


def is_exam_instruction(text):
    return bool(EXAM_NOTE.match(text))


def find_answer_boundary(pages):
    for index,page in enumerate(pages,1):
        for line in page['lines']:
            text=re.sub(r'\s+','',line['text'])
            if len(text)<=100 and re.fullmatch(r'.{0,80}(?:参考答案(?:与试题解析|及评分标准|及解析|与解析)?|答案与解析|答案及解析|参考解答|试题解析|答案详解|答案及评分标准|听力原文|听力材料|录音原文|听力录音稿).{0,8}',text):
                return dict(page=index,y=line['box'][1],x=line['box'][0],method='native_answer_heading')
            if re.fullmatch(r'答案(?:(?:及|与|和)?(?:解析|详解))?',text):
                return dict(page=index,y=line['box'][1],x=line['box'][0],method='native_answer_heading')
            if re.fullmatch(r'.{0,80}(?:数学|英语|试卷|试题|考试|测试|练习)(?:参考)?答案(?:(?:及|与|和)(?:解析|评分标准|评分细则))?',text):
                return dict(page=index,y=line['box'][1],x=line['box'][0],method='native_answer_heading')
    for index,page in enumerate(pages,1):
        rows=[l for l in page['lines'] if re.fullmatch(r'[A-G](?:\s+[A-G]){3,}',l['text'].strip())]
        if len(rows)>=2 and sum(len(l['text'].split()) for l in rows)>=8:
            first=min(rows,key=lambda l:l['box'][1])
            return dict(page=index,y=first['box'][1],x=first['box'][0],method='native_answer_grid')
    for index,page in enumerate(pages,1):
        for line in page['lines']:
            if SOLUTION_CUE.search(line['text']):
                before=[int(s['number']) for i,p in enumerate(pages,1) for s in p['question_starts']
                        if not s.get('instruction') and (i<index or (i==index and s['box'][1]<line['box'][1]))]
                after=[int(s['number']) for i,p in enumerate(pages,1) for s in p['question_starts']
                       if not s.get('instruction') and (i>index or (i==index and s['box'][1]>line['box'][1]))]
                interleaved=before and after and after[0]>before[-1] and all(b>a for a,b in zip(before,before[1:]))
                if (index==1 and before) or interleaved:
                    return dict(page=len(pages)+1,y=0,method='native_interleaved_answers')
                return dict(page=index,y=line['box'][1],x=line['box'][0],method='native_solution_start')
    return dict(page=len(pages)+1,y=0,method='native_question_only_no_solution_markers')


def native_pages(path):
    pages = []
    # WPS PDFs can emit XML-forbidden control glyphs; their coordinates and other words remain usable.
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]", "", Path(path).read_text())
    for element in ET.fromstring(raw).findall(".//x:page", NS):
        width, height = float(element.get("width")), float(element.get("height"))
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("invalid PDF page dimensions")
        lines = []
        for line in element.findall(".//x:line", NS):
            box = [float(line.get(k)) for k in ("xMin", "yMin", "xMax", "yMax")]
            if not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= width + 2 and 0 <= box[1] < box[3] <= height + 2):
                continue
            words = line.findall("x:word", NS)
            text = " ".join(w.text or "" for w in words)
            item=dict(box=box,text=text)
            if words and re.match(r'\s*(?:\d{1,3}|[（(]\d{1,3}[）)])',words[0].text or ''):
                try:
                    wb=[float(words[0].get(k)) for k in ('xMin','yMin','xMax','yMax')]
                    if all(math.isfinite(v) for v in wb) and 0<=wb[0]<wb[2]<=width+2 and 0<=wb[1]<wb[3]<=height+2:item['number_box']=wb
                except (TypeError,ValueError):pass
            lines.append(item)
        lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
        # ponytail: exclude margin-only text; 30 body characters is a conservative routing rule, not a crop-quality claim.
        body = [l for l in lines if .03 * height <= l["box"][1] < .92 * height and
                not re.fullmatch(r"第\s*\d+\s*页\s*[/／]?\s*共\s*\d+\s*页", l["text"].strip())]
        chars = sum(len(re.findall(r"[\w\u4e00-\u9fff]", l["text"])) for l in body)
        footers=[l['box'][1]-2 for l in lines if l['box'][1]>.85*height and
                 (re.search(r'第\s*\d+\s*页.{0,8}共\s*\d+\s*页',l['text']) or re.fullmatch(r'\s*\d+\s*[/／]\s*\d+\s*',l['text']))]
        body_bottom=min(footers) if footers else height*.97
        anchor_lines=[l for l in lines if .03*height<=l['box'][1]<body_bottom]
        starts = []
        for line in anchor_lines:
            match = re.match(r"\s*(\d{1,3})\s*(?:[.．、](?!\d)|\s+(?=假[设如定]|你校|你是|你的|请你))", line["text"])
            parenthesized=re.match(r'\s*[（(](\d{1,3})[）)]',line['text'])
            if not match and parenthesized and Path(path).stem.startswith('english-'):
                match=parenthesized
            if match and line["box"][0] < .22 * width:
                anchor_box=line.get('number_box',line['box'])
                same_row=sorted((l for l in anchor_lines if (l is line or abs(l['box'][1]-anchor_box[1])<2.5) and l['box'][0]>=anchor_box[0]-1),key=lambda l:l['box'][0])
                anchor_text=' '.join(l['text'] for l in same_row)
                starts.append(dict(number=match[1],box=anchor_box,text=anchor_text,instruction=is_exam_instruction(anchor_text)))
        pages.append(dict(width=width, height=height, lines=lines, body_characters=chars,body_bottom=body_bottom,
                          native_coordinates=chars >= 30 or any(not s['instruction'] for s in starts), question_starts=starts))
    if not pages:
        raise ValueError("no PDF pages extracted")
    return pages


def fetch_preview(paper, token):
    path = DIRECTORY / "pdfs" / (("math" if paper["subject"] == "数学" else "english") + "-" + paper["id"] + ".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 5:
        with path.open("rb") as stream:
            if stream.read(5) == b"%PDF-":
                return path
    if not token:
        raise FileNotFoundError('cached PDF unavailable; offline audit does not make network requests')
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.jingshibang.com/home/",
               "Authorization": "Bearer " + token, "Authori-zation": "Bearer " + token}
    def request(url):
        # Credentials travel only over stdin, never argv, logs, progress files or exception URLs.
        config = "url = " + json.dumps(url) + "\n" + "".join("header = " + json.dumps(k + ": " + v) + "\n" for k, v in headers.items())
        result = subprocess.run(["curl", "--noproxy", "*", "--silent", "--connect-timeout", "10", "--max-time", "40",
                                 "--max-filesize", "100000000", "--write-out", "\n%{http_code}", "--config", "-"],
                                input=config.encode(), capture_output=True, timeout=45)
        payload, _, status = result.stdout.rpartition(b"\n")
        if status in (b"401", b"403", b"429"):
            STOP.set()
            raise PermissionError("preview access needs review; network requests paused")
        if result.returncode or status != b"200":
            raise OSError("preview transport failed")
        time.sleep(.1)
        return payload
    for attempt in range(3):
        if STOP.is_set():
            raise PermissionError("preview access paused")
        try:
            detail = json.loads(request("https://www.jingshibang.com/api/product/detail/" + paper["id"]))
            if detail.get("status") != 200:
                STOP.set()
                raise PermissionError("preview session needs review")
            store = detail["data"]["storeInfo"]
            filepath = store.get("pdf_answer") or store.get("pdf_paper")
            if not filepath:
                raise ValueError("no normal PDF preview")
            params = dict(product_id=paper["id"], type="answer" if store.get("pdf_answer") else "paper", filepath=filepath, token=token)
            data = request("https://www.jingshibang.com/api/public/filepreview?" + urlencode(params))
            if not data.startswith(b"%PDF-"):
                raise ValueError("normal preview did not return a PDF")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(path)
            return path
        except (OSError, subprocess.TimeoutExpired):
            if STOP.is_set() or attempt == 2:
                raise
            time.sleep(2 ** attempt)


def inspect(paper, token, refresh=False):
    key = ("math" if paper["subject"] == "数学" else "english") + "-" + paper["id"]
    state_path = DIRECTORY / "papers" / (key + ".json")
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        unchanged_heading=saved.get('version')=='native-v2' and saved.get('answer_boundary',{}).get('method')=='native_answer_heading'
        if not refresh and saved.get("state") == "checked" and (saved.get("version") == AUDIT_VERSION or unchanged_heading):
            return saved
    started = time.monotonic()
    result = dict(key=key, subject=paper["subject"], grade=paper["grade"], paper_id=paper["id"],
                  title=paper["title"], source_pages=paper["pages"], version=AUDIT_VERSION)
    for field in ('original_source_pages','original_source_folder','source_folder','preview_revision'):
        if field in paper:result[field]=paper[field]
    try:
        pdf = fetch_preview(paper, token)
        xml = pdf.with_suffix(".html")
        if not xml.exists():
            temporary = xml.with_suffix(".html.part")
            subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(temporary)], check=True, capture_output=True, timeout=90)
            temporary.replace(xml)
        pages = native_pages(xml)
        if len(pages) != paper["pages"]:
            raise ValueError("PDF page count changed from saved screenshots")
        native = sum(p["native_coordinates"] for p in pages)
        boundary = find_answer_boundary(pages)
        native_question_pages = sum(p['native_coordinates'] and
                                    (i<boundary['page'] or (i==boundary['page'] and any(s['box'][1]<boundary['y'] and not s.get('instruction') for s in p['question_starts'])))
                                    for i,p in enumerate(pages,1)) if boundary else None
        result.update(state="checked", native_pages=native, non_native_pages=len(pages) - native,
                      layout="all_native" if native == len(pages) else "mixed" if native else "no_native",
                      answer_boundary=boundary, native_question_pages=native_question_pages,
                      pages=[dict(page=i, native_coordinates=p["native_coordinates"], body_characters=p["body_characters"],
                                  question_starts=len(p["question_starts"])) for i, p in enumerate(pages, 1)],
                      pdf=str(pdf.relative_to(ROOT)), seconds=round(time.monotonic() - started, 3))
    except Exception as error:
        result.update(state="pending", error=type(error).__name__ + ": " + str(error)[:100].replace(token, "[redacted]"))
    atomic_json(state_path, result)
    return result


def audit(auth_file=None, sample=0):
    import random
    papers = load_papers()
    population = collections.Counter((p["subject"], p["grade"]) for p in papers)
    if sample:
        random.seed(20261026)
        papers = [p for group in sorted(population) for p in random.sample([p for p in papers if (p["subject"], p["grade"]) == group], sample // len(population))]
    else:
        random.Random(20261026).shuffle(papers)
    token = json.loads(Path(auth_file).read_text())["token"] if auth_file else ''
    started = time.time()
    progress = dict(state="running", started=started, pid=__import__("os").getpid(), planned_papers=len(papers),
                    planned_pages=sum(p["pages"] for p in papers), checked_papers=0, checked_pages=0,
                    native_papers=0, all_native_papers=0, mixed_papers=0, no_native_papers=0,
                    native_pages=0, non_native_pages=0,native_question_pages=0, pending_papers=0, strata={}, sample=bool(sample))
    atomic_json(DIRECTORY / "audit-manifest.json", dict(papers=papers, population={str(k):v for k,v in population.items()}))
    # ponytail: two threads preserve the capture job's existing network concurrency; cached PDFs are reused by later cropping.
    current=[]
    with (DIRECTORY / "audit-records.jsonl").open("a", encoding="utf-8") as journal, concurrent.futures.ThreadPoolExecutor(max_workers=2 if token else 4) as pool:
        futures = [pool.submit(inspect, p, token) for p in papers]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            current.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            if result["state"] == "checked":
                progress["checked_papers"] += 1
                progress["checked_pages"] += result["source_pages"]
                progress["native_pages"] += result["native_pages"]
                progress['native_question_pages']+=result.get('native_question_pages') or 0
                progress["non_native_pages"] += result["non_native_pages"]
                progress["native_papers"] += result["native_pages"] > 0
                progress[result["layout"] + "_papers"] += 1
                group = result["subject"] + " " + result["grade"]
                row = progress["strata"].setdefault(group, dict(papers=0, native_papers=0, pages=0, native_pages=0))
                row["papers"] += 1; row["native_papers"] += result["native_pages"] > 0
                row["pages"] += result["source_pages"]; row["native_pages"] += result["native_pages"]
            else:
                progress["pending_papers"] += 1
            progress.update(updated=time.time(), elapsed_seconds=round(time.time() - started, 2), network_paused=STOP.is_set())
            atomic_json(DIRECTORY / "audit-progress.json", progress)
            if progress["checked_papers"] % 20 == 0:
                print("NATIVE_AUDIT", json.dumps(progress, ensure_ascii=False), flush=True)
            if STOP.is_set():
                for pending in futures:
                    pending.cancel()
                break
    progress.update(state="checked" if progress["checked_papers"] == len(papers) else "incomplete", updated=time.time())
    if len(current)==len(papers) and not sample:
        temporary=DIRECTORY/'audit-current.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in current))
        temporary.replace(DIRECTORY/'audit-current.jsonl')
    atomic_json(DIRECTORY / "audit-progress.json", progress)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-file",help='省略时只复用已缓存 PDF，不访问网络')
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (DIRECTORY / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("AUDIT_FINISHED", json.dumps(audit(args.auth_file, args.sample), ensure_ascii=False), flush=True)
