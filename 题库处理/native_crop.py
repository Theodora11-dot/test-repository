"""Crop single questions from audited PDF coordinates and the original JPEGs; no OCR."""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import time

from PIL import Image
import numpy as np

from pipeline import ROOT, atomic_json, image_path, load_papers, pin_image, source_dimensions
from pdf_native import DIRECTORY, native_pages, is_exam_instruction, SOLUTION_CUE
from embedding import question_skip_reason

RUN = DIRECTORY / "crops-run"
HEADING = re.compile(r"^(?:第[一二三四五六七八九十\d]+[部分节章]|[一二三四五六七八九十]+[、．.]|(?:书面表达|应用文写作|读后续写|语法填空|七选五)(?:[（(:：\s]|$))")
VERSION = "native-crops-v2"
ALGORITHM = "native-coordinates-v7"
RECHECK_VERSION = "word-anchors-footers-essays-material-v2"
DATA_RECHECK = 'english-source-integrity-v1'
WRITING_START=re.compile(r'假[设如定]|你校|你的|你是|请(?:你|根据|以)|Suppose|\bWrite\b|[（(]\s*20\d{2}',re.I)
SOURCE_LABEL=re.compile(r'^[（(]\s*(?:19|20)\d{2}[^()（）\n]{0,100}(?:高[一二三]|高考|校考|统考|学业考试|期中|期末|月考|模拟|开学考)[^()（）\n]{0,50}[）)]')


def source_row_top(lines,label):
    center=(label['box'][1]+label['box'][3])/2
    tolerance=max(3,(label['box'][3]-label['box'][1])*.6)
    return min(l['box'][1] for l in lines if abs((l['box'][1]+l['box'][3])/2-center)<=tolerance)


def separator(ink_rows, anchor, scale, minimum=0, max_ink=2):
    """Snap a native baseline to whitespace, retaining formula ascenders and descenders."""
    low, high = max(0,round(anchor-20*scale),round(minimum)), min(len(ink_rows),round(anchor)+1)
    blank = ink_rows[low:high] <= max_ink
    edges = np.diff(np.r_[False,blank,False].astype(int))
    gaps = [(int(b-a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=3]
    if not gaps:
        return None
    length,end = max(gaps)
    return low+end-length//2


def body_lines(page):
    return [l for l in page["lines"] if .025 * page["height"] < l["box"][1] < page.get('body_bottom',.935*page['height'])
            and not re.search(r"第\s*\d+\s*页.{0,5}共\s*\d+\s*页", l["text"])]


def plans(paper, pages, boundary):
    """Return native source rectangles for single questions; continuations remain deferred."""
    found, deferred = [], []
    reading = any(t in paper["title"] for t in ("阅读", "完形", "七选五"))
    writing_carry=any(t in paper['title'] for t in ('写作','作文','书面表达','读后续写'))
    parent_carry=None
    for index, page in enumerate(pages, 1):
        if boundary and index > boundary["page"]:
            break
        if not page["native_coordinates"]:
            continue
        width, height = page["width"], page["height"]
        end = boundary["y"] if boundary and index == boundary["page"] else page.get('body_bottom',height*.935)
        lines = [l for l in body_lines(page) if l["box"][1] < end]
        starts = [s for s in page["question_starts"] if s["box"][1] < end]
        starts.sort(key=lambda s: (s["box"][1], s["box"][0]))
        # Same baseline fragments can describe the same question number.
        starts = [s for i,s in enumerate(starts) if not i or abs(s["box"][1]-starts[i-1]["box"][1]) > 3]
        writing,writing_parent=writing_carry,parent_carry
        for start in starts:
            headings=[l for l in lines if l['box'][1]<start['box'][1] and HEADING.match(l['text'])]
            if headings:
                last=headings[-1]['text']
                if any(t in last for t in ('写作','作文','书面表达','应用文','读后续写')):writing=True
                elif any(t in last for t in ('阅读','填空','翻译','单项','单词','听力')):writing=False
            if paper['subject']=='英语' and writing_parent is None:
                floor=headings[-1]['box'][3] if headings else 0
                preface='\n'.join(l['text'] for l in lines if floor<=l['box'][1]<start['box'][1])
                if re.search(r'(?m)^假[设如定]你',preface) and re.search(r'写|邮件|作文',preface):
                    # Unnumbered writing prompts own their numbered bullet notes too.
                    writing=True;writing_parent=999
            text=start.get('text','')
            real_prompt=bool(WRITING_START.search(text))
            start['nested']=bool(writing and writing_parent is not None and not real_prompt)
            if writing and not start.get('instruction') and not start['nested']:
                writing_parent=start['number']
        writing_carry,parent_carry=writing,writing_parent
        cloze_options=paper['subject']=='英语' and any(re.match(r'^\s*(?:\d{1,3}[.．、]|[（(]\d{1,3}[）)])\s*A[.．、]',s.get('text','')) for s in starts)
        page_text=''.join(l['text'] for l in lines)
        gap_selection=paper['subject']=='英语' and bool(re.search(r'(?:五|七|5|7)个选项',page_text) and re.search(r'空白|填入',page_text))
        if cloze_options or gap_selection:reading=True
        if paper['subject']=='英语' and ('完形' in paper['title'] or cloze_options or gap_selection):
            for start in starts:
                # Inline (3) blanks belong to the article; only (3) A. ... is an option row.
                if re.match(r'^\s*[（(]\s*\d{1,3}\s*[）)](?!\s*[A-D][.．、])',start.get('text','')):
                    start['nested']=True
        synthetic=[]
        for line in lines:
            text=line['text'].strip()
            prompt=(paper['subject']=='英语' and bool(re.match(r'^(?:假[设如定]你|Suppose you|你校(?:将|计划)|你是|你的英国|你的美国)',text,re.I)))
            chinese_math=(paper['subject']=='数学' and bool(re.match(r'^(?:第[一二三四五六七八九十\d]+题|[一二三四五六七八九十]+[、．.]|[【\[]?例题?\s*[一二三四五六七八九十\d]+)',text))
                          and bool(re.search(r'已知|如图|设|求|证明',text))
                          and (not re.search(r'解答应|证明过程|演算步骤|本大题|每小题',text) or bool(re.search(r'已知|如图',text))))
            if not (prompt or chinese_math):continue
            previous=[s for s in starts if not s.get('instruction') and not s.get('nested') and 0<=line['box'][1]-s['box'][1]<50]
            if previous:continue
            synthetic.append(dict(number='未编号题'+str(len(synthetic)+1),box=line['box'],text=text,number_source='native_prompt'))
        section_headers=[l for l in lines if HEADING.match(l['text'])]
        for hi,header in enumerate(section_headers):
            kind=('grammar_passage' if '语法填空' in header['text'] else 'gap_selection' if '七选五' in header['text'] else
                  'writing' if any(t in header['text'] for t in ('书面表达','应用文写作','读后续写')) else None)
            if not kind:continue
            floor=header['box'][3];ceiling=section_headers[hi+1]['box'][1] if hi+1<len(section_headers) else end
            existing=[s for s in starts+synthetic if not s.get('instruction') and not s.get('nested') and floor<=s['box'][1]<ceiling]
            material=[l for l in lines if floor<=l['box'][1]<ceiling and l['text'].strip()]
            if existing or not material:continue
            text='\n'.join(l['text'] for l in material)
            if kind!='writing' and len(re.findall(r'\b[A-Za-z]+\b',text))<40:continue
            if kind=='writing' and not re.search(r'假[设如定]|Suppose|\bWrite\b|请.{0,20}写',text,re.I):continue
            synthetic.append(dict(number=header['text'].strip()+'（未编号）',box=material[0]['box'],text=material[0]['text'],
                                  number_source='native_prompt',native_task=kind))
        if synthetic:
            starts=sorted(starts+synthetic,key=lambda s:(s['box'][1],s['box'][0]))
        active=[s for s in starts if not s.get('instruction') and not s.get('nested')]
        context, context_text = None, ""
        prefix_continuation=False
        if active and paper["subject"] == "英语":
            before = [l for l in lines if (l['box'][1]+l['box'][3])/2 < active[0]["box"][1] - 2]
            heading = [l for l in before if HEADING.match(l["text"]) or re.fullmatch(r"[A-D]", l["text"].strip())]
            if any("阅读" in l["text"] or "完形" in l["text"] for l in heading):
                reading = True
            floor = max((l["box"][3] for l in heading), default=0)
            citations=[l for l in before if SOURCE_LABEL.match(l['text'].strip()) and '汇编' not in l['text']]
            if citations:floor=max(floor,source_row_top(before,citations[-1]))
            material = [l for l in before if (l['box'][1]+l['box'][3])/2 >= floor]
            material_text = "\n".join(l["text"] for l in material)
            if heading and len(re.findall(r'\b[A-Za-z]+\b',material_text))>=40:reading=True
            prefix_continuation=index>1 and not heading and len(re.findall(r'\b[A-Za-z]+\b',material_text))>=40
            span=active[0]['box'][1]-floor
            citation_material=citations and len(re.findall(r'\b[A-Za-z]+\b',material_text))>=40
            if material and (reading or not re.search(r'[?？]',material_text)) and (len(re.findall(r"\b[A-Za-z]+\b", material_text)) >= 8 or (reading and span>60)) and (index == 1 or heading or citation_material):
                context = [width * .035, min(l["box"][1] for l in material) - 2, width * .965, active[0]["box"][1] - 3]
                context_text = material_text
        for position, start in enumerate(starts):
            if start.get('instruction') or start.get('nested'):
                deferred.append(dict(page=index,number=start['number'],reason='non_question_instruction'));continue
            number = start["number"]
            top = start["box"][1] - 3
            following_starts=[s for s in active if s['box'][1]>start['box'][1]+3]
            next_start=following_starts[0] if following_starts else None
            bottom = next_start["box"][1] - 3 if next_start else end - 3
            section = [l for l in lines if start['box'][1]+3 < l["box"][1] < bottom and HEADING.match(l["text"])]
            if section:
                bottom = min(l["box"][1] for l in section) - .5
            cues=[l for l in lines if top<l['box'][1]<bottom and SOLUTION_CUE.search(l['text'])
                  and ((boundary or {}).get('method')=='native_interleaved_answers' or not re.match(r'^\s*提示\s*[:：]',l['text']))]
            if cues:bottom=min(l['box'][1] for l in cues)-.5
            selected = [l for l in lines if top <= (l['box'][1]+l['box'][3])/2 < bottom]
            text = "\n".join(l["text"] for l in selected)
            next_context, next_text = None, ""
            prefix_missing=prefix_continuation
            if paper["subject"] == "英语":
                latest_heading=None
                for line in lines:
                    if line["box"][1] < top and HEADING.match(line["text"]):
                        latest_heading=line
                        if any(t in line["text"] for t in ("阅读", "完形")):reading = True
                        elif any(t in line["text"] for t in ("语法", "写作", "翻译", "单项",'单词','完成句子','选词','词汇','听力','书面表达')):reading = False
                if latest_heading and context and latest_heading['box'][1]>context[1]:
                    context,context_text=None,''
                if latest_heading and latest_heading['box'][1]>active[0]['box'][1]:prefix_missing=False
                if latest_heading and not context:
                    floor=latest_heading['box'][3]
                    material=[l for l in lines if floor<=l['box'][1]<top]
                    mt='\n'.join(l['text'] for l in material)
                    prior_questions=[s for s in active if floor<s['box'][1]<top]
                    if material and not prior_questions and (reading or not re.search(r'[?？]',mt)) and (len(re.findall(r'\b[A-Za-z]+\b',mt))>=8 or (reading and top-floor>60)):
                        context=[width*.035,min(l['box'][1] for l in material)-2,width*.965,top];context_text=mt
                citations=[l for l in selected if l['box'][1]>start['box'][1]+18 and SOURCE_LABEL.match(l['text'].strip()) and '汇编' not in l['text']]
                if citations:
                    cut=source_row_top(selected,citations[0])
                    previous='\n'.join(l['text'] for l in selected if (l['box'][1]+l['box'][3])/2<cut)
                    tail=[l for l in selected if (l['box'][1]+l['box'][3])/2>=cut]
                    tail_text='\n'.join(l['text'] for l in tail)
                    table_words=sum(bool(re.fullmatch(r'\s*[A-Za-z]+\s*',l['text'])) for l in tail)
                    word_bank=not reading and (bool(re.search(r'选词|短语|方框',tail_text)) or table_words>=3) and len(re.findall(r'\b[A-Za-z]+\b',tail_text))>=5
                    if len(re.findall(r'\b[A-Za-z]+\b',previous))>=3 and (len(re.findall(r'\b[A-Za-z]+\b',tail_text))>=40 or word_bank):
                        next_context=[width*.035,cut-2,width*.965,bottom];next_text=tail_text
                        bottom=cut-2;selected=[l for l in selected if (l['box'][1]+l['box'][3])/2<bottom];text='\n'.join(l['text'] for l in selected)
                choices = [l for l in selected if re.search(r"(?:^|\s)D[.．、]", l["text"])]
                if choices and next_context is None:
                    last = max(l["box"][3] for l in choices)
                    tail = [l for l in selected if l["box"][1] > last + 2]
                    tail_text = "\n".join(l["text"] for l in tail)
                    if tail and (len(re.findall(r"\b[A-Za-z]+\b", tail_text)) >= 40 or tail[-1]['box'][3]-tail[0]['box'][1]>60):
                        next_context = [width*.035, tail[0]["box"][1]-2, width*.965, bottom]
                        next_text = tail_text
                        bottom = tail[0]["box"][1] - 4
                        text = "\n".join(l["text"] for l in selected if l["box"][1] < bottom)
                has_choices = all(re.search(r"(?:^|\s)" + c + r"[.．、]", text) for c in "ABCD")
                if reading and not has_choices and len(re.findall(r"\b[A-Za-z]+\b", text)) >= 100:
                    # A numbered passage is context, not an independent multiple-choice item.
                    context, context_text = [width*.035, top, width*.965, bottom], text
                    deferred.append(dict(page=index, number=number, reason="material_without_its_questions"))
                    continue
            reason = None
            if not cues and next_start is None and index < len(pages) and not (boundary and boundary["page"] <= index):
                following = pages[index]
                following_end=boundary['y'] if boundary and boundary['page']==index+1 else following.get('body_bottom',following['height']*.935)
                following_lines = [l for l in body_lines(following) if not HEADING.match(l["text"]) and
                                   l['box'][1]<following_end and not re.fullmatch(r"[A-D]", l["text"].strip()) and len(l["text"].strip()) >= 2]
                next_starts = [s for s in following["question_starts"] if not s.get('instruction') and s['box'][1]<following_end
                               and (not writing or WRITING_START.search(s.get('text','')))]
                if (not following_lines and following_end>following['height']*.2) or (following_lines and
                        (not next_starts or min(s["box"][1] for s in next_starts) > min(l["box"][1] for l in following_lines)+3)):
                    reason = "cross_page"
            reference_question=bool(re.search(r'\b(?:passage|paragraph|writer|author|according to|main idea|best title|inferred|(?:the|this)\s+text)\b',text,re.I)) and not start.get('native_task')
            embedded=bool(start.get('native_task')) or (len(re.findall(r'\b[A-Za-z]+\b',text))>=100 and all(re.search(r'\b'+c+r'[.．、]',text) for c in 'ABCD'))
            if (reading or reference_question or prefix_missing) and paper["subject"] == "英语" and not context and not embedded:
                reason = "shared_material_unconfirmed"
            boxes = ([context] if context else []) + [[width*.035, top, width*.965, bottom]]
            question_text = (context_text + "\n" if context else "") + text
            candidate = dict(subject=paper["subject"], title=paper["title"], page=index, number=number,
                             role="question", status="complete", text=question_text,
                             number_source=start.get('number_source','printed'),
                             native_task=start.get('native_task'),
                             native_context_state='same_page_material' if context else 'embedded_material' if embedded else 'standalone')
            reason = reason or question_skip_reason(candidate)
            if reason:
                deferred.append(dict(page=index, number=number, reason=reason))
            elif bottom > top:
                found.append(dict(page=index, number=number, text=question_text, boxes=boxes, page_size=[width,height], ordinal=position+1,
                                  anchor_box=start['box'],native_context_state=candidate['native_context_state']))
                found[-1]['number_source']=candidate['number_source']
                found[-1]['native_task']=candidate['native_task']
            if next_context:
                context, context_text = next_context, next_text
    return found, deferred


def crop_paper(paper, audit, refresh=False):
    key = audit["key"]
    done = RUN / "papers" / (key + ".json")
    previous=json.loads(done.read_text()) if done.exists() else {}
    if not refresh and previous.get('algorithm_revision')==ALGORITHM:
        return dict(previous,events=[] if previous.get('journaled_revision')==ALGORITHM else previous.get('events',previous['records']))
    old_records={r['id']:r for r in previous.get('records',[])}
    used=set()
    pages = native_pages(ROOT / audit["pdf"].replace(".pdf", ".html"))
    boundary = audit.get("answer_boundary")
    if not boundary:
        return dict(key=key, state="review", reason="answer_boundary_unconfirmed", records=[], source_pages=paper["pages"])
    candidates, deferred = plans(paper, pages, boundary)
    target = DIRECTORY / "crops" / paper["subject"] / paper["id"]
    target.mkdir(parents=True, exist_ok=True)
    records = []
    image, current_page = None, None
    try:
        for item in candidates:
            if item["page"] != current_page:
                if image is not None:image.close()
                current_page = item["page"]
                with Image.open(image_path(paper,current_page)) as source:image=source.convert("RGB")
                if abs(image.width/image.height - item["page_size"][0]/item["page_size"][1]) > .003:
                    raise ValueError("source screenshot and PDF aspect ratios differ")
                gray=np.asarray(image.convert('L'))
                ink_rows=(gray[:,round(image.width*.035):round(image.width*.965)]<160).sum(1)
            sx,sy=image.width/item["page_size"][0],image.height/item["page_size"][1]
            boxes = [[max(0,round(x1*sx)),max(0,round(y1*sy)),min(image.width,round(x2*sx)),min(image.height,round(y2*sy))]
                     for x1,y1,x2,y2 in item["boxes"]]
            anchors=[s for s in pages[current_page-1]['question_starts'] if not s.get('instruction') and not s.get('nested')]
            unclear=False
            for source_box,pixel_box in zip(item['boxes'],boxes):
                for edge in (1,3):
                    nearby=[a for a in anchors if abs(source_box[edge]-(a['box'][1]-3))<1.1]
                    if nearby:
                        anchor=nearby[0]['box']
                        preceding=[l['box'][3] for l in body_lines(pages[current_page-1])
                                   if l['box'][1]<anchor[1]-3 and l['box'][3]<anchor[1] and l['box'][0]<=anchor[0]+4]
                        minimum=(max(preceding)-.5)*sy if preceding else 0
                        snapped=separator(ink_rows,anchor[1]*sy,sy,minimum)
                        if snapped is None:
                            # ponytail: tolerate thin table rules only when no clear whitespace separator exists.
                            snapped=separator(ink_rows,anchor[1]*sy,sy,minimum,max_ink=max(2,round(image.width*.004)))
                        if snapped is None:unclear=True
                        else:pixel_box[edge]=snapped
            for box in boxes:
                occupied=np.flatnonzero(ink_rows[box[1]:box[3]]>2)
                if occupied.size and box[3]-(box[1]+occupied[-1])>100:
                    box[3]=min(box[3],int(box[1]+occupied[-1]+12))
            text='\n'.join(line['text'] for box in boxes for line in pages[current_page-1]['lines']
                           if box[1]<=(line['box'][1]+line['box'][3])/2*sy<box[3])
            if unclear or not text.strip() or any(box[3]-box[1]<12 for box in boxes) or question_skip_reason(dict(subject=paper['subject'],title=paper['title'],
                    page=item['page'],number=item['number'],role='question',status='complete',text=text,
                    number_source=item['number_source'],native_task=item['native_task'],native_context_state=item['native_context_state'])):
                deferred.append(dict(page=item['page'],number=item['number'],reason='crop_needs_review'));continue
            matching=[r for r in old_records.values() if r['id'] not in used and r['page']==item['page'] and r['number']==item['number']]
            old=min(matching,key=lambda r:abs(r['source_boxes'][-1][1]-boxes[-1][1])) if matching else None
            qid=old['id'] if old else f"{key}-p{item['page']:03d}-q{20000+item['ordinal']:05d}"
            while qid in used or (not old and qid in old_records):
                qid=qid.rsplit('-q',1)[0]+'-q'+str(int(qid.rsplit('-q',1)[1])+1)
            used.add(qid)
            pieces=[image.crop(box) for box in boxes]
            if len(pieces)==1:
                picture=pieces[0];bbox=boxes[0];space="source_image"
            else:
                picture=Image.new("RGB",(max(p.width for p in pieces),sum(p.height for p in pieces)+12*(len(pieces)-1)),"white")
                y=0
                for piece in pieces:picture.paste(piece,(0,y));y+=piece.height+12
                bbox=[0,0,picture.width,picture.height];space="composite_image"
            same_image=old and old.get('source_boxes')==boxes and old.get('source_image')==str(image_path(paper,item['page']))
            if same_image:
                crop=ROOT/old['crop_file'];asset=old['image']
            else:
                stat=image_path(paper,item['page']).stat()
                recipe=json.dumps([str(image_path(paper,item['page'])),stat.st_size,stat.st_mtime_ns,boxes])
                asset_id=qid+'-'+hashlib.sha256(recipe.encode()).hexdigest()[:12]
                crop=target/(asset_id+'.jpg');temporary=crop.with_suffix('.jpg.tmp')
                picture.save(temporary,format="JPEG",quality=95);temporary.replace(crop)
                asset=str((ROOT/'native-assets'/key/pin_image(crop,asset_id,root=ROOT/'native-assets'/key)).relative_to(ROOT))
            record=dict(id=qid,paper_id=paper["id"],subject=paper["subject"],title=paper["title"],number=item["number"],page=item["page"],
                        source_image=str(image_path(paper,item["page"])),bbox=bbox,source_boxes=boxes,coordinate_space=space,
                        image=asset,
                        crop_file=str(crop.relative_to(ROOT)),text=text,
                        dimensions=source_dimensions(paper),status="complete",role="question",crop_method="native_pdf_coordinates",
                        native_answer_boundary=dict(boundary, source_y=round(boundary['y']*sy)),
                        source_url="https://www.jingshibang.com/home/#/paperDetail?id="+paper["id"],version=VERSION,
                        algorithm_revision=ALGORITHM,number_source=item['number_source'],native_task=item['native_task'],
                        native_context_state=item['native_context_state'])
            if paper.get('preview_revision'):
                record.update(preview_revision=paper['preview_revision'],original_source_pages=paper['original_source_pages'])
            atomic_json(crop.with_suffix('.json'),record);records.append(record)
    finally:
        if image is not None:image.close()
    events=[r for r in records if r!=old_records.get(r['id'])]
    events.extend(dict(r,role='other',status='review',retired_by=ALGORITHM) for qid,r in old_records.items() if qid not in used)
    result=dict(key=key,state="cropped" if records else "review",version=VERSION,algorithm_revision=ALGORITHM,
                coordinate_review_revision=RECHECK_VERSION,records=records,events=events,deferred=deferred,source_pages=paper["pages"])
    if paper['subject']=='英语':result['data_review_revision']=DATA_RECHECK
    atomic_json(done,result)
    return result


def run(follow=True):
    RUN.mkdir(parents=True,exist_ok=True)
    papers={(p["subject"],p["id"]):p for p in load_papers()}
    owners={}
    for shard in range(8):
        for paper in json.loads((ROOT/'run'/f'shard-{shard}.json').read_text()):owners[(paper['subject'],paper['id'])]=shard
    states=[dict(shard=i,state="running",pid=os.getpid(),processed_papers=0,cropped_questions=0,review_papers=0,failed_papers=0) for i in range(8)]
    seen=set();offset=0
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    # PDF parsing is Python CPU work: separate processes bypass the GIL; each paper owns its files.
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={}
        while True:
            audit_state=json.loads((DIRECTORY/'audit-progress.json').read_text())
            final=audit_state['state']!='running'
            manifest=DIRECTORY/'audit-current.jsonl'
            with (manifest if manifest.exists() else DIRECTORY/'audit-records.jsonl').open() as source:
                source.seek(offset)
                while True:
                    line=source.readline()
                    if not line.endswith('\n'):break
                    offset=source.tell()
                    try:audit=json.loads(line)
                    except ValueError:continue
                    if audit['key'] in seen or audit.get('state')!='checked':continue
                    seen.add(audit['key']);paper=papers[(audit['subject'],audit['paper_id'])]
                    if audit.get('source_folder'):
                        paper=dict(paper,folder=audit['source_folder'],pages=audit['source_pages'],
                                   original_source_pages=audit['original_source_pages'],preview_revision=audit['preview_revision'])
                    pending[pool.submit(crop_paper,paper,audit)]=(paper,audit)
            complete=[f for f in pending if f.done()]
            for future in complete:
                paper,audit=pending.pop(future);shard=owners[(paper['subject'],paper['id'])];state=states[shard]
                try:
                    result=future.result()
                    with (RUN/f'results-{shard}.jsonl').open('a',encoding='utf-8') as journal:
                        for record in result.get('events',result['records']):journal.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if result.get('algorithm_revision')==ALGORITHM and result.get('events'):
                        result['journaled_revision']=ALGORITHM
                        atomic_json(RUN/'papers'/(audit['key']+'.json'),result)
                    state['processed_papers']+=1;state['cropped_questions']+=len(result['records'])
                    state['review_papers']+=result['state']!='cropped'
                except Exception as error:
                    state['failed_papers']+=1
                    with (RUN/f'errors-{shard}.jsonl').open('a') as errors:errors.write(json.dumps(dict(key=audit['key'],error=repr(error)),ensure_ascii=False)+'\n')
                print('NATIVE_CROPS',audit['key'],state['cropped_questions'],flush=True)
            for i,state in enumerate(states):
                state.update(updated=time.time(),state='finished' if final and not pending else 'running')
                atomic_json(RUN/f'progress-{i}.json',state)
            atomic_json(RUN/'status.json',dict(state='finished' if final and not pending else 'running',
                        cropped_questions=sum(s['cropped_questions'] for s in states),processed_papers=sum(s['processed_papers'] for s in states),
                        failed_papers=sum(s['failed_papers'] for s in states),workers=states))
            if (final and not pending) or not follow:break
            time.sleep(3)


if __name__=='__main__':
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
