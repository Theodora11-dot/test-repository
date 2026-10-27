"""Evaluate fixed natural-language prompts, or run software checks with --check."""
import argparse
import html
import json
import subprocess
import sys
import time

import numpy as np

from common import ROOT, atomic_json
from embedding import Encoder, MODEL, MAX_PIXELS, check_encoder
sys.path.insert(0, str(ROOT.parent))
from infer import open_index, prepare_search, rank

DIRECTORY = ROOT / 'embedding-native-questions'


def demo(db, encoder, queries=None, name="evaluation-current", vectors=None,device=None):
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
                  note=("来源主题检查仅用于试运行，不代表逐题准确率；按原生 PDF 中的答案边界排除解析页；"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', nargs='?', default='evaluation-current', help='评估报告文件名，不含扩展名')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--holdout', action='store_true', help='使用另外 10 组提示词')
    group.add_argument('--all-tests', action='store_true', help='使用全部 20 组提示词')
    group.add_argument('--check', action='store_true', help='运行软件检查，不加载模型或真实题库')
    args = parser.parse_args()
    if args.check:
        for command in [('test_index.py',), ('test_pdf_native.py',), ('native_coverage.py', '--check'), ('prepare.py', '--check')]:
            subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:]], check=True)
        subprocess.run([sys.executable, str(ROOT.parent / '北京题库/test_config.py')], check=True)
        return
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('报告名称只能包含英文字母、数字、下划线和连字符')
    prompt_names = ['evaluation-prompts-holdout.json' if args.holdout else 'evaluation-prompts.json']
    vector_names = ['evaluation-holdout-queries.npy' if args.holdout else 'evaluation-initial-queries.npy']
    if args.all_tests:
        prompt_names.append('evaluation-prompts-holdout.json')
        vector_names.append('evaluation-holdout-queries.npy')
    queries = [query for name in prompt_names for query in json.loads((DIRECTORY / name).read_text())['queries']]
    # ponytail: retain the existing cached prompt vectors; fresh installs encode the fixed prompts once.
    vectors = (np.concatenate([np.load(DIRECTORY / name, allow_pickle=False) for name in vector_names])
               if all((DIRECTORY / name).exists() for name in vector_names) else None)
    db = open_index(DIRECTORY / 'index.sqlite3')
    try:
        check_encoder(db)
        demo(db, Encoder(MAX_PIXELS) if vectors is None else None, queries, args.name, vectors)
    finally:
        db.close()


if __name__ == '__main__':
    main()
