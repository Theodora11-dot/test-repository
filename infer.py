"""Query an existing question index using natural language and return screenshot records."""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Keep the shared encoder and data helpers beside the preparation code.
sys.path.insert(0, str(Path(__file__).resolve().parent / '题库处理'))
from common import ROOT, question_skip_reason
from embedding import Encoder, MAX_PIXELS, check_encoder, unit_vector

INDEX = ROOT / 'embedding-native-questions/index.sqlite3'

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


def prepare_search(db,device=None):
    """Load one immutable search snapshot once for a batch of queries."""
    # ponytail: scan SQLite sequentially on network disks, then restore ID order in memory for stable ties.
    rows = db.execute("SELECT id,record,vector FROM documents NOT INDEXED").fetchall()
    rows.sort(key=lambda row: row[0])
    rows = [(record, vector) for _, record, vector in rows]
    paper_rows = db.execute("SELECT id,record,vector FROM papers").fetchall()
    papers = {row[0]: (json.loads(row[1]), unit_vector(np.frombuffer(row[2], dtype="<f2"))) for row in paper_rows}
    question_index=db.execute("SELECT value FROM settings WHERE key='unit'").fetchone()==('question',)
    # ponytail: exact cosine search fits the current 186k questions; use ANN if measured latency requires it.
    # Avoid constructing and stacking a separate NumPy view for each of the 186k vectors.
    if rows and (not rows[0][1] or len(rows[0][1]) % 2 or len({len(row[1]) for row in rows}) != 1):
        raise ValueError('索引向量为空或长度不一致')
    matrix = (np.frombuffer(bytearray().join(row[1] for row in rows), dtype='<f2').reshape(len(rows), -1)
              if rows else np.empty((0, 0), dtype='<f2'))
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
                 if records[i].get("image")]
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


def open_index(path=INDEX):
    # Query connections never create a database, tables, checkpoints or lock files.
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def search(db, encoder, query, limit=5, corpus=None):
    if not query.strip():
        return []
    if len(query) > 4096:
        raise ValueError('查询过长，最多 4096 个字符')
    if not 1 <= limit <= 100:
        raise ValueError('limit 必须在 1 到 100 之间')
    vector = encoder.encode(texts=[query])[0]
    return rank(db, vector, limit, query_text=query,
                corpus=corpus if corpus is not None else prepare_search(db, device=encoder.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='自然语言找题要求')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--index', type=Path, default=INDEX)
    parser.add_argument('--max-pixels', type=int, default=MAX_PIXELS)
    args = parser.parse_args()
    if len(args.query) > 4096 or not 1 <= args.limit <= 100:
        parser.error('查询最多 4096 个字符，limit 必须在 1 到 100 之间')
    if not args.query.strip():
        print('[]')
        return
    db = open_index(args.index)
    try:
        check_encoder(db, args.max_pixels)
        results = search(db, Encoder(args.max_pixels), args.query, args.limit)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == '__main__':
    main()
