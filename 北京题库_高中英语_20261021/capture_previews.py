"""把当前账号正常可见的 PDF 预览逐页保存为图片；保留页面原貌。"""

import argparse
import csv
import json
import signal
from http.client import HTTPException
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parent
BASE = "https://www.jingshibang.com"
NETWORK = threading.Semaphore(2)
STOP = threading.Event()


def jpeg_ok(path):
    with path.open("rb") as stream:
        first = stream.read(2)
        stream.seek(-2, 2)
        return first == b"\xff\xd8" and stream.read(2) == b"\xff\xd9"


def main(auth_file, limit):
    token = json.loads(Path(auth_file).read_text())["token"]
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    headers = {"User-Agent": "Mozilla/5.0", "Referer": BASE + "/home/", "Authorization": "Bearer " + token, "Authori-zation": "Bearer " + token}
    snapshot = json.loads((ROOT / "原始目录.json").read_text(encoding="utf-8"))
    items = {}
    for query in snapshot["queries"]:
        for item in query["items"]:
            items.setdefault(item["id"], {**item, "grade": query["grade"]})
    queue = sorted(items.values(), key=lambda item: (item.get("add_time", ""), item["id"]), reverse=True)
    if limit:
        queue = queue[:limit]
    records = {}
    journal = ROOT / "预览图片记录.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record

    def request(path, params=None, binary=False):
        if STOP.is_set():
            raise PermissionError("会话已停止")
        url = BASE + "/api" + path + ("?" + urlencode(params) if params else "")
        for attempt in range(3):
            if STOP.is_set():
                raise PermissionError("会话已停止")
            try:
                with NETWORK:
                    # ponytail: curl提供含DNS及整个传输的总时限；凭证仅经标准输入传入，不跟随重定向。
                    config = "url = " + json.dumps(url) + "\n" + "".join("header = " + json.dumps(k + ": " + v) + "\n" for k, v in headers.items())
                    try:
                        result = subprocess.run(["curl", "--silent", "--show-error", "--noproxy", "*", "--connect-timeout", "10", "--max-time", "30", "--max-filesize", "100000000", "--write-out", "\n%{http_code}", "--config", "-"], input=config.encode(), capture_output=True, timeout=35)
                    except subprocess.TimeoutExpired:
                        raise TimeoutError("网络请求超过总时限") from None
                    if result.returncode:
                        raise OSError("网络传输失败，curl退出码" + str(result.returncode))
                    payload, _, status = result.stdout.rpartition(b"\n")
                    status = int(status)
                    if status != 200:
                        raise HTTPError("", status, "官网响应错误", None, None)
                    data = payload if binary else json.loads(payload)
                    time.sleep(0.2)
                break
            except (OSError, HTTPException) as error:
                if getattr(error, "code", None) in (401, 403, 429) or isinstance(error, PermissionError):
                    STOP.set()
                if STOP.is_set() or attempt == 2:
                    raise RuntimeError(type(error).__name__) from None
                time.sleep(2 ** attempt)
        if binary:
            if len(data) > 100_000_000:
                raise ValueError("预览超过100MB，需单独处理")
            if not data.startswith(b"%PDF-"):
                raise ValueError("官网未返回PDF预览")
            return data
        if data.get("status") != 200:
            message = str(data.get("msg", "官网未允许访问")).replace(token, "[redacted]")
            if "登录" in message or data.get("status") in (401, 403, 429):
                STOP.set()
            raise PermissionError(message[:100])
        return data["data"]

    def capture(item):
        record = {"id": item["id"], "grade": item["grade"], "title": item["store_name"], "price": item.get("price"), "pages": 0, "status": "未处理", "path": ""}
        previous = records.get(item["id"], {})
        if previous.get("status") == "已保存":
            folder = ROOT / previous["path"]
            if (folder / "页数说明.json").exists() and len(list(folder.glob("page-*.jpg"))) == previous["pages"]:
                return previous
        title = re.sub(r"[\\/\x00-\x1f]", "_", item["store_name"])[:65]
        parent = ROOT / "预览图片" / item["grade"]
        target = parent / (str(item["id"]) + "_" + title)
        if (target / "页数说明.json").exists():
            recovered = json.loads((target / "页数说明.json").read_text(encoding="utf-8"))
            if recovered["id"] == item["id"] and len(list(target.glob("page-*.jpg"))) == recovered["pages"]:
                return recovered
        work = None
        try:
            fetch_started = time.monotonic()
            detail = request("/product/detail/" + str(item["id"]))["storeInfo"]
            filepath = detail.get("pdf_answer") or detail.get("pdf_paper")
            if not filepath:
                record["status"] = "无PDF预览，需单独检查网页或网盘"
                return record
            data = request("/public/filepreview", {"product_id": item["id"], "type": "answer" if detail.get("pdf_answer") else "paper", "filepath": filepath, "token": token}, binary=True)
            fetch_seconds = round(time.monotonic() - fetch_started, 2)
            render_started = time.monotonic()
            parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("目标目录已存在但未通过断点核验，需人工检查")
            work = Path(tempfile.mkdtemp(prefix=".pages-" + str(item["id"]) + "-", dir=parent))
            with tempfile.TemporaryDirectory(prefix="jsb-preview-render-") as private:
                pdf = Path(private) / "preview.pdf"
                pdf.write_bytes(data)
                info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=60, check=True)
                pages = int(re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)[1])
                subprocess.run(["pdftoppm", "-scale-to", "2200", "-jpeg", "-jpegopt", "quality=94", str(pdf), str(work / "page")], capture_output=True, timeout=600, check=True)
            images = sorted(work.glob("page-*.jpg"))
            if len(images) != pages or not all(jpeg_ok(path) for path in images):
                raise ValueError("图片页数或JPEG完整性核验失败")
            record.update(fetch_seconds=fetch_seconds, render_seconds=round(time.monotonic() - render_started, 2), pages=pages, status="已保存", path=str(target.relative_to(ROOT)), bytes=sum(p.stat().st_size for p in images), source=BASE + "/home/#/paperDetail?id=" + str(item["id"]), note="官网正常预览返回的页面；未逐份核对原出版页数")
            (work / "页数说明.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            work.rename(target)
        except Exception as error:
            record["status"] = "待处理"
            record["reason"] = (str(error) if isinstance(error, (PermissionError, ValueError, RuntimeError)) else type(error).__name__).replace(token, "[redacted]")[:180]
        finally:
            if work and work.exists():
                shutil.rmtree(work)
        return record

    started = time.monotonic()
    with journal.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(capture, item): item["id"] for item in queue}
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records[record["id"]] = record
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            counts = Counter(r["status"] for r in records.values())
            progress = {"planned": len(queue), "checked": number, "saved": counts["已保存"], "pages": sum(r["pages"] for r in records.values()), "status_counts": dict(counts), "stopped_for_access": STOP.is_set(), "elapsed_seconds": round(time.monotonic() - started, 1)}
            temporary = ROOT / "预览图片进度.json.tmp"
            temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(ROOT / "预览图片进度.json")
            if number <= 6 or number % 25 == 0 or number == len(queue):
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            if STOP.is_set():
                for pending in futures:
                    pending.cancel()
                break
    with (ROOT / "预览图片清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["id", "grade", "title", "price", "pages", "status", "path", "reason"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records.values())
    print("CAPTURE_FINISHED", json.dumps(progress, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-file")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        records = {}
        for line in (ROOT / "预览图片记录.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[record["id"]] = record
        saved = [r for r in records.values() if r["status"] == "已保存"]
        assert saved, "尚无已完成图片"
        for record in saved:
            folder = ROOT / record["path"]
            images = list(folder.glob("page-*.jpg"))
            assert len(images) == record["pages"] > 0 and all(jpeg_ok(path) for path in images), record["id"]
        print("核验通过：", len(saved), "份预览的图片页数、JPEG首尾和清单一致。")
    else:
        if not args.auth_file:
            parser.error("需要 --auth-file 指向已授权登录会话的私有凭证文件")
        main(args.auth_file, args.limit)
