"""导出菁师帮公开的高中数学资料目录；不下载文件、不消耗账号额度。"""

import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
BASE = "https://www.jingshibang.com"
GRADES = ("高一", "高二", "高三")
FIELDS = ["资料ID", "年级", "标题", "年份", "区县", "类型", "目录来源", "标价（元）", "答案标注", "上传日期", "详情链接"]


def fetch(endpoint, params):
    request = Request(BASE + "/api" + endpoint + "?" + urlencode(params), headers={
        "User-Agent": "Mozilla/5.0", "Referer": BASE + "/home/",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=40) as response:
                result = json.load(response)
            if result.get("status") != 200:
                raise ValueError(f"官网返回错误：{result.get('msg')}")
            return result["data"]
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def check(items, total):
    if len(items) != total:
        raise ValueError(f"目录不完整：期望 {total}，实际 {len(items)}")
    ids = [item["id"] for item in items]
    if len(set(ids)) != total or any(not isinstance(i, int) or i <= 0 for i in ids):
        raise ValueError("资料ID存在重复或异常")
    if any(item.get("store_subject") != "数学" for item in items):
        raise ValueError("目录混入非数学资料")


def write_csv(filename, rows):
    # ponytail: 用标准库 CSV，Excel 可直接打开；不引入电子表格依赖。
    with (ROOT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for row in rows:
            writer.writerow(["'" + str(v) if str(v).lstrip().startswith(("=", "+", "-", "@")) else v for v in row])


def main(from_snapshot=False):
    started = datetime.now().astimezone().isoformat()
    snapshot = {"started_at": started, "source": BASE, "complete": False, "queries": []}
    merged = {}
    queries = [(g, "/products", {"store_grade": g, "include_assembly": 1}) for g in GRADES]
    cached = json.loads((ROOT / "原始目录.json").read_text(encoding="utf-8"))["queries"] if from_snapshot else []
    for grade, endpoint, filters in queries:
        items, total, page = [], None, 1
        previous = next((q for q in cached if q["grade"] == grade and q["endpoint"] == endpoint and q["filters"] == filters), None)
        if previous:
            items, total = previous["items"], previous["total"]
        while total is None or len(items) < total:
            params = {"store_subject": "数学", "page": page, "limit": 100, **filters}
            result = fetch(endpoint, params)
            batch = result
            count = int(batch[0].get("count", 0)) if batch else 0
            if total is None:
                total = count
            elif count and count != total:
                raise ValueError(f"{grade}采集期间目录数量发生变化，请重试")
            if not batch and len(items) < total:
                raise ValueError(f"{grade}第{page}页意外为空")
            items.extend(batch)
            print(f"{grade} {endpoint}: {len(items)}/{total}", flush=True)
            page += 1
            time.sleep(0.25)
        check(items, total)
        snapshot["queries"].append({"grade": grade, "endpoint": endpoint, "filters": filters, "total": total, "items": items})
        temporary = ROOT / "原始目录.json.tmp"
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(ROOT / "原始目录.json")
        for item in items:
            entry = merged.setdefault(item["id"], {"item": item, "grades": set(), "sources": set()})
            if grade in GRADES:
                entry["grades"].add(grade)
            entry["sources"].add("试卷与汇编")

    rows = []
    for identifier, entry in merged.items():
        item = entry["item"]
        title = item["store_name"]
        grades = entry["grades"] or set(re.findall(r"高[一二三]", title)) or {"高中通用"}
        year = re.match(r"\d{4}(?:[-—]\d{4})?", title)
        rows.append([
            identifier, "、".join(sorted(grades)), title, item.get("store_year") or (year.group() if year else ""),
            item.get("district", ""), item.get("store_type") or item.get("paper_type", ""),
            "、".join(sorted(entry["sources"])), item.get("price", ""), item.get("have_answer", "未标注"),
            item.get("add_time", ""), BASE + "/home/#/paperDetail?" + urlencode({"id": identifier, "title": title}),
        ])
    rows.sort(key=lambda row: (row[1], row[3], row[2], row[0]), reverse=True)
    write_csv("高中数学_全部目录.csv", rows)
    for grade in GRADES:
        write_csv(grade + "数学_目录.csv", [row for row in rows if grade in row[1].split("、")])
    (ROOT / "数学资料_搜索标题.txt").write_text("\n".join(sorted({row[2] for row in rows}, reverse=True)) + "\n", encoding="utf-8")
    snapshot.update(complete=True, completed_at=datetime.now().astimezone().isoformat(), unique_count=len(rows))
    (ROOT / "原始目录.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "北京题库（菁师帮）高中数学资料目录", "采集时间：" + started,
        "范围：官网高一、高二、高三数学筛选结果，所有年份，包含其中的试卷与汇编。",
        "各目录分页结果已核对总数、唯一资料ID、数学学科；按资料ID合并重复记录。",
        *[f"{q['grade']} {q['endpoint']}：{q['total']} 条" for q in snapshot["queries"]],
        f"合并后共 {len(rows)} 份资料。",
        "归档年级：" + json.dumps(dict(Counter(row[1] for row in rows)), ensure_ascii=False),
        "年级为官网筛选结果。搜索清单保存的是官网资料标题，不是服务器上的原始上传文件名。",
        "CSV 可用 Excel 打开；详情链接需要在官网中登录后使用。",
        "这是资料目录，CSV 与 JSON 本身不含试题正文。答案未标注不代表没有答案。",
        "标价及会员下载权限以官网当前状态为准；本文档不包含账号凭证。",
        "重新采集：python3 collect_catalog.py；离线核验：python3 collect_catalog.py --check",
    ]
    (ROOT / "采集说明.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("CATALOG_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        saved = json.loads((ROOT / "原始目录.json").read_text(encoding="utf-8"))
        assert saved["complete"]
        for query in saved["queries"]:
            check(query["items"], query["total"])
        identifiers = {i["id"] for q in saved["queries"] for i in q["items"]}
        with (ROOT / "高中数学_全部目录.csv").open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == len(identifiers) == saved["unique_count"]
        assert {int(row["资料ID"]) for row in exported} == identifiers
        print("核验通过：分页总数、ID唯一性、数学学科、CSV/JSON记录一致。")
    else:
        main(from_snapshot="--from-snapshot" in sys.argv[1:])
