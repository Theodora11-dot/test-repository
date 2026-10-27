"""读取外站公开目录，以规范化同名匹配；不会请求受限下载。"""

import argparse
import csv
import hashlib
import html
import json
import re
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, urlopen

from collect_catalog import check
from settings import DEFAULT_CONFIG, load_config

CACHE = Path("/tmp/jsb-public-catalog-cache")
SITE = "https://gaokaotiku.com"


def normalize(title):
    title = unicodedata.normalize("NFKC", html.unescape(title))
    title = re.sub(r"\.(?:pdf|docx?)$", "", title, flags=re.I)
    title = re.sub(r"\((?:教师版|学生版|含答案|有答案)\)", "", title)
    title = re.sub(r"[-_](?:试题|答案)$", "", title)
    return re.sub(r"[\s\W_]+", "", title, flags=re.UNICODE)


def get_page(url):
    url = quote(url, safe=":/?=&%")
    cache = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".html")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=20) as response:
                content = response.read(2_000_000).decode("utf-8", "replace")
            if "<h2" not in content or "高考题库" not in content:
                raise ValueError("页面未返回预期目录")
            cache.write_text(content, encoding="utf-8")
            time.sleep(0.35)
            return content
        except (OSError, ValueError):
            if attempt:
                raise
            time.sleep(1)


def entries(url, content, subject):
    # ponytail: 仅适配已核验的 h2 目录模板；模板变化时改用 HTMLParser。
    result = []
    for href, label in re.findall(r'<h2\b[^>]*>\s*<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>', content, re.S):
        target = urljoin(url, html.unescape(href))
        title = html.unescape(re.sub(r"<[^>]+>", "", label)).strip()
        if urlsplit(target).hostname == "gaokaotiku.com" and subject in title:
            result.append({"title": title, "url": target})
    return result


def main(subject, root, grades):
    root = Path(root)
    CACHE.mkdir(exist_ok=True)
    snapshot = json.loads((root / "原始目录.json").read_text(encoding="utf-8"))
    originals = {}
    lookup = defaultdict(list)
    for query in snapshot["queries"]:
        check(query['items'], query['total'], subject)
        if query['grade'] not in grades:
            continue
        for item in query["items"]:
            if item["id"] not in originals:
                originals[item["id"]] = {**item, "grade": query["grade"]}
                lookup[normalize(item["store_name"])].append(item["id"])
    sources, tasks, failures = {}, [], []
    for grade in grades:
        base = SITE + "/a/" + subject + "/" + grade + "/"
        page = get_page(base)
        for item in entries(base, page, subject):
            sources[item["url"]] = item
        last = re.search(r"href=['\"](list_\d+_(\d+)\.html)['\"][^>]*>末页", page)
        if not last:
            raise ValueError("外站分页模板发生变化：" + grade)
        pattern = last[1].rsplit("_", 1)[0] + "_{}.html"
        tasks.extend(urljoin(base, pattern.format(number)) for number in range(2, int(last[2]) + 1))
        print(grade, "外站标注", re.findall(r"\d+页\d+条", page), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(get_page, url): url for url in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                for item in entries(url, future.result(), subject):
                    sources[item["url"]] = item
            except (OSError, ValueError) as error:
                failures.append({"url": url, "error": type(error).__name__})
            if number % 20 == 0 or number == len(tasks):
                print("目录页", number, "/", len(tasks), "外站条目", len(sources), "失败页", len(failures), flush=True)
    matches = defaultdict(list)
    for source in sources.values():
        for identifier in lookup.get(normalize(source["title"]), []):
            matches[identifier].append(source)
    with (root / "外站同名检索.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["资料ID", "年级", "原始标题", "外站标题", "外站页面", "匹配方式", "下载状态", "说明"])
        for identifier, item in originals.items():
            found = matches.get(identifier, [])
            for source in found or [{}]:
                writer.writerow([
                    identifier, item["grade"], item["store_name"], source.get("title", ""), source.get("url", ""),
                    "规范化同名" if source else "该站目录未匹配；其他网站待查",
                    "该站下载抽查要求权限；本条未逐一验证" if source else "待检索",
                    "外站标注答案时可能只有答案，不能视为完整试卷；内容尚未逐页比对。" if source else "",
                ])
    evidence = {
        "source": SITE, "original_count": len(originals), "external_count": len(sources),
        "matched_originals": len(matches), "matched_links": sum(map(len, matches.values())),
        "failed_pages": failures, "sources": list(sources.values()),
    }
    (root / "外站目录证据.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MATCHING_COMPLETE", {key: evidence[key] for key in ("original_count", "external_count", "matched_originals", "matched_links")}, "failed_pages", len(failures), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check:
        assert normalize("2025北京昌平高一（上）期末英语（教师版）-答案") == normalize("2025北京昌平高一（上）期末英语")
        assert normalize("2025北京昌平高一（上）期末英语") != normalize("2025北京昌平高一（下）期末英语")
        assert normalize("2025北京昌平高一（上）期末英语") != normalize("2026北京昌平高一（上）期末英语")
        for subject in config['subjects']:
            sample = f'<h2 class="x"><a href="/a/{subject}/1.html">2025北京高一{subject}（上）</a></h2>'
            assert entries(SITE, sample, subject) == [{"title": f"2025北京高一{subject}（上）", "url": SITE + f"/a/{subject}/1.html"}]
        print("核验通过：全半角与版本标记可匹配，年份和上下学期不会混淆。")
    else:
        for subject, root in config['subjects'].items():
            main(subject, root, config['grades'])
