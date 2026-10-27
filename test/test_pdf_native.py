"""Run: python test/test_pdf_native.py. No network or GPU."""
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
from common import ROOT
from pdf_native import native_pages,find_answer_boundary,is_exam_instruction


def check():
    words = "1．已知函数的图像如下，请计算它的周期并判断单调区间，在指定区间求出最大值和最小值。"
    with tempfile.TemporaryDirectory() as temporary:
        xml = Path(temporary) / "pages.html"
        xml.write_text('<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>'
                       '<page width="600" height="840"><flow><block><line xMin="50" yMin="100" xMax="550" yMax="115">'
                       '<word>' + words + '</word></line></block></flow></page>'
                       '<page width="600" height="840"><flow><block><line xMin="250" yMin="800" xMax="350" yMax="812">'
                       '<word>第 2 页/共 2 页</word></line></block></flow></page></doc></body></html>')
        pages = native_pages(xml)
        assert len(pages) == 2 and pages[0]["native_coordinates"]
        assert pages[0]["question_starts"][0]["number"] == "1"
        assert not pages[1]["native_coordinates"], "footer-only scanned pages cannot be counted as native body text"
        xml.write_text(xml.read_text().replace('已知','\x01已知'))
        assert native_pages(xml)[0]['native_coordinates'], 'invalid control glyphs must not lose an entire paper'
        sample=lambda text:dict(lines=[dict(text=text,box=[40,100,500,115])],question_starts=[])
        for heading in ['参考答案与试题解析','高二英语参考答案','答案及解析','北京一零一中2023-2024学年度第一学期期中考试高一数学参考答案','听力原文']:
            assert find_answer_boundary([sample('试题'),sample(heading)])['page']==2
        assert find_answer_boundary([sample('一道没有答案的试题')])['method']=='native_question_only_no_solution_markers'
        assert find_answer_boundary([sample('【答案】A')])['method']=='native_solution_start'
        interleaved=sample('【答案】A');interleaved['question_starts']=[dict(number='1',box=[40,80,500,95])]
        assert find_answer_boundary([interleaved])['method']=='native_interleaved_answers'
        assert is_exam_instruction('1. 本试卷共六页')
        word_xml=Path(temporary)/'word-anchors.html'
        word_xml.write_text('<html xmlns="http://www.w3.org/1999/xhtml"><body><doc><page width="600" height="840">'
            '<flow><block><line xMin="50" yMin="100" xMax="550" yMax="160">'
            '<word xMin="50" yMin="140" xMax="65" yMax="155">1．</word>'
            '<word xMin="70" yMin="100" xMax="550" yMax="160">已知函数的图像和方程，求它的性质。</word>'
            '</line></block></flow></page></doc></body></html>')
        assert native_pages(word_xml)[0]['question_starts'][0]['box'][1]==140, 'oversized formula boxes must not move the printed question number upward'
        word_xml.write_text('<html xmlns="http://www.w3.org/1999/xhtml"><body><doc><page width="600" height="840"><flow><block>'
            '<line xMin="50" yMin="790" xMax="500" yMax="805"><word>12. What does the speaker mean?</word></line>'
            '<line xMin="280" yMin="825" xMax="350" yMax="835"><word>第 1 页/共 2 页</word></line>'
            '</block></flow></page></doc></body></html>')
        page=native_pages(word_xml)[0]
        assert page['question_starts'][0]['number']=='12' and page['body_bottom']==823, 'real bottom-of-page questions must precede the detected footer rather than a fixed margin'
    root = ROOT
    digital = root / "pdf-fast/pdfs/math-195356.html"
    scanned = root / "pdf-fast/pdfs/math-199003.html"
    if digital.exists() and scanned.exists():
        assert all(p["native_coordinates"] for p in native_pages(digital))
        assert not any(p["native_coordinates"] for p in native_pages(scanned))
        from native_crop import plans
        from common import load_papers
        papers={p['id']:p for p in load_papers() if p['subject']=='英语'}
        for pid,page_number,opening in [('81840',13,'One day I read an article'),('83707',1,'I was ready to pay for my bananas')]:
            cached=native_pages(root/'pdf-fast/pdfs'/('english-'+pid+'.html'))
            proposed,_=plans(papers[pid],cached,find_answer_boundary(cached))
            questions=[q for q in proposed if q['page']==page_number]
            assert questions and all(opening in q['text'] for q in questions), 'inline cloze gap numbers must not replace the article opening'
        cached=native_pages(root/'pdf-fast/pdfs/english-83716.html')
        proposed,_=plans(papers['83716'],cached,find_answer_boundary(cached))
        assert not any(q['page']==6 for q in proposed), 'a continuation page cannot supply its own complete cloze article'
        for pid,first_answer in [('170140',12),('171622',8),('162580',11)]:
            cached=native_pages(root/'pdf-fast/pdfs'/('english-'+pid+'.html'))
            boundary=find_answer_boundary(cached)
            assert boundary['page']==first_answer, 'unlabelled answer grids and long school answer headings must bound the question region'
            proposed,_=plans(papers[pid],cached,boundary)
            assert not any(q['page']>=first_answer for q in proposed)
        for pid,pn,number in [('106146',4,'6'),('83262',7,'第三部分'),('81840',4,None),('102229',2,None),('54108',14,None)]:
            cached=native_pages(root/'pdf-fast/pdfs'/('english-'+pid+'.html'))
            proposed,_=plans(papers[pid],cached,find_answer_boundary(cached))
            assert not any(q['page']==pn and (number is None or str(q['number']).startswith(number)) for q in proposed), 'missing articles, continuation pages and title-only tasks cannot become complete questions'
        cached=native_pages(root/'pdf-fast/pdfs/english-125574.html')
        proposed,_=plans(papers['125574'],cached,find_answer_boundary(cached))
        banks={q['number']:q['text'] for q in proposed if q['page']==1}
        assert 'bring up' in banks['8'] and 'keep up' not in banks['8'], 'a new source word bank must replace the previous bank'
        assert 'confuse' in banks['16'] and 'bring up' not in banks['16'], 'a word table without its own instruction still starts a new word bank'
        cached=native_pages(root/'pdf-fast/pdfs/english-171417.html')
        proposed,_=plans(papers['171417'],cached,find_answer_boundary(cached))
        assert all('Did you know the man' not in q['text'] for q in proposed if q['page']==7), 'a sample question is not shared material'
        from native_crop import separator
        import numpy as np
        ink=np.ones(80,dtype=int)*10;ink[25:35]=0
        assert 25<separator(ink,45,1)<35
        assert separator(np.ones(80,dtype=int)*10,45,1) is None
        ink=np.ones(80,dtype=int)*10;ink[25:35]=0;ink[40:44]=0
        assert separator(ink,45,1,minimum=39)==42, 'a previous blank line must not erase a dense question'
        from common import load_papers
        papers=load_papers()
        paper=next(p for p in papers if p['subject']=='数学' and p['id']=='195356')
        found,deferred=plans(paper,native_pages(digital),dict(page=3,y=52.19))
        assert [(q['page'],q['number']) for q in found]==[(1,str(i)) for i in range(1,6)]
        assert any(q['number']=='6' and q['reason']=='cross_page' for q in deferred)
        english=root/'pdf-fast/pdfs/english-195063.html'
        if english.exists():
            paper=next(p for p in papers if p['subject']=='英语' and p['id']=='195063')
            found,deferred=plans(paper,native_pages(english),dict(page=3,y=52.628))
            assert len(found)==17 and not deferred
        writing=dict(width=600,height=840,native_coordinates=True,lines=[
            dict(text='书面表达',box=[40,100,100,112]),dict(text='假设你是李华，请给外国朋友写一封邮件。',box=[40,125,500,137]),
            dict(text='1. 介绍自己的爱好；',box=[40,150,500,162]),dict(text='2. 希望了解对方更多信息。',box=[40,175,500,187])],
            question_starts=[dict(number='1',text='1. 介绍自己的爱好；',box=[40,150,500,162]),dict(number='2',text='2. 希望了解对方更多信息。',box=[40,175,500,187])])
        found,deferred=plans(dict(subject='英语',title='英语考试'),[writing],dict(page=2,y=0))
        assert len(found)==1 and found[0]['number_source']=='native_prompt' and '2. 希望' in found[0]['text'], 'writing bullet notes belong to the unnumbered parent prompt'
        import concurrent.futures,multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            parallel=pool.submit(plans,dict(subject='英语',title='英语考试'),[writing],dict(page=2,y=0)).result()
        assert parallel==(found,deferred), 'separate CPU workers must preserve the crop plan'
        intro=dict(text='1. 假设你是李华，请给朋友写一封邮件，内容包括：',box=[40,700,500,712])
        first=dict(width=600,height=840,native_coordinates=True,lines=[intro],question_starts=[dict(number='1',**intro)])
        following=dict(width=600,height=840,native_coordinates=True,lines=writing['lines'][2:],question_starts=writing['question_starts'])
        found,deferred=plans(dict(subject='英语',title='书信写作'),[first,following],dict(page=3,y=0))
        assert not found and any(d['reason']=='cross_page' for d in deferred), 'writing notes on the following page must not become standalone questions'
        unnumbered=dict(width=600,height=840,native_coordinates=True,question_starts=[],lines=[dict(text='一、已知a>0，求证下述结论并给出证明过程。',box=[40,100,500,115])])
        found,_=plans(dict(subject='数学',title='数学'),[unnumbered],dict(page=2,y=0))
        assert len(found)==1 and '已知' in found[0]['text'] and found[0]['boxes'][-1][3]>115, 'a real unnumbered problem must not be cut at its own heading'
        unnumbered['lines'][0]['text']='三、解答题，共85分。解答应写出演算步骤或证明过程。'
        found,_=plans(dict(subject='数学',title='数学'),[unnumbered],dict(page=2,y=0))
        assert not found, 'section instructions are not mathematical questions'
        prompt=dict(text='1. 假设你是李华，请给朋友写一封邀请信。',box=[40,100,500,115])
        gpt=dict(width=600,height=840,native_coordinates=True,question_starts=[dict(number='1',**prompt)],lines=[prompt,
            dict(text='【作者 ChatGPT】',box=[40,200,160,215]),dict(text='Dear Jim, I hope this email finds you well.',box=[40,230,500,245])])
        found,_=plans(dict(subject='英语',title='ChatGPT版写作'),[gpt],dict(page=2,y=0,method='native_answer_heading'))
        assert len(found)==1 and 'I hope' not in found[0]['text'] and found[0]['boxes'][-1][3]<200, 'inline model essays must be clipped even when a later answer section exists'
        material=dict(width=600,height=840,native_coordinates=True,question_starts=[dict(number='2',text='2. What did the writer discover?',box=[40,400,500,415])],lines=[
            dict(text='（2024 北京高考真题）A story',box=[40,60,300,75]),
            dict(text='A short story about people helping others. '*15,box=[40,100,550,360]),
            dict(text='2. What did the writer discover?',box=[40,400,500,415]),
            dict(text='A. Help. B. Rest. C. Sleep. D. Leave.',box=[40,425,550,440])])
        empty=dict(width=600,height=840,native_coordinates=False,question_starts=[],lines=[])
        found,_=plans(dict(subject='英语',title='阅读理解'),[empty,material],dict(page=3,y=0))
        assert len(found)==1 and len(found[0]['boxes'])==2 and 'people helping' in found[0]['text'], 'an explicit exam source label proves a same-page article start without an A/B/C heading'
    print("PASS native coordinates, question anchors, footer exclusion, digital/scanned calibration")


if __name__ == "__main__":
    check()
