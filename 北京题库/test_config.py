"""Run: python 北京题库/test_config.py. No network, credentials, model or GPU needed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import yaml

import capture_previews as capture
import collect_catalog as catalog
import find_same_titles as matching
from settings import ROOT, load_config

sys.path.insert(0, str(ROOT/'题库处理'))
from common import load_papers, image_path


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root/'config/config.yaml'
        config_path.parent.mkdir()
        raw = dict(subjects={'数学':'../data/math', '英语':'../data/english'}, grades=['高一'])
        def save(value):
            config_path.write_text(yaml.safe_dump(value, allow_unicode=True))
        save(raw)
        previous = Path.cwd()
        os.chdir(root)
        try:
            config = load_config(config_path)
            assert config['subjects']['数学'] == root/'data/math'
            assert config['subjects']['英语'] == root/'data/english'
            for invalid in [None, dict(raw, grades='高一'), dict(raw, grades=[]),
                            dict(raw, grades=['高一','高一']), dict(raw, grade=['高一']),
                            dict(raw, subjects={'物理':'../data/physics'}),
                            dict(raw, subjects={'数学':'../same', '英语':'../same'}),
                            dict(raw, subjects={'数学':str(ROOT/'北京题库')}),
                            dict(raw, subjects={'数学':''})]:
                save(invalid)
                try:
                    load_config(config_path)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid configuration accepted')
            save(raw)

            calls = []
            def fake_fetch(endpoint, params):
                calls.append(params['store_subject'])
                subject = params['store_subject']
                return [dict(id=7, count=1, store_subject=subject, store_name=f'2025北京高一{subject}')]
            original_fetch, original_sleep = catalog.fetch, catalog.time.sleep
            catalog.fetch, catalog.time.sleep = fake_fetch, lambda _:None
            try:
                for subject, directory in config['subjects'].items():
                    catalog.main(subject, directory, config['grades'])
                    if subject == '数学':
                        snapshot_path = directory/'原始目录.json'
                        snapshot = json.loads(snapshot_path.read_text())
                        snapshot['queries'].append(dict(grade='高二', endpoint='/products',
                            filters=dict(store_grade='高二', include_assembly=1), total=1,
                            items=[dict(id=8, count=1, store_subject=subject, store_name='2025北京高二数学')]))
                        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False))
                    catalog.main(subject, directory, config['grades'], from_snapshot=True)
                    assert (directory/f'高中{subject}_全部目录.csv').exists()
                    if subject == '数学':
                        snapshot = json.loads(snapshot_path.read_text())
                        assert snapshot['unique_count'] == 2 and {q['grade'] for q in snapshot['queries']} == {'高一','高二'}, 'grade filtering must preserve already collected source records'
                assert calls == ['数学','英语'], 'both subjects must use one collector; replay must not fetch again'
            finally:
                catalog.fetch, catalog.time.sleep = original_fetch, original_sleep

            requests = []
            def fake_run(command, **kwargs):
                if command[0] == 'curl':
                    config_text = kwargs['input'].decode()
                    url = json.loads(config_text.splitlines()[0].split(' = ',1)[1])
                    requests.append(urlsplit(url).path)
                    if '/product/detail/' in url:
                        payload = json.dumps(dict(status=200,data=dict(storeInfo=dict(pdf_paper='preview.pdf')))).encode()
                    else:
                        payload = b'%PDF-test'
                    return subprocess.CompletedProcess(command, 0, stdout=payload+b'\n200', stderr=b'')
                if command[0] == 'pdfinfo':
                    return subprocess.CompletedProcess(command, 0, stdout='Pages: 1\n', stderr='')
                assert command[0] == 'pdftoppm'
                Path(command[-1]+'-1.jpg').write_bytes(b'\xff\xd8\xff\xd9')
                return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')
            auth = root/'session.json'
            auth.write_text(json.dumps(dict(token='synthetic-test-token')))
            original_run, original_sleep = capture.subprocess.run, capture.time.sleep
            original_signal = signal.getsignal(signal.SIGTERM)
            capture.subprocess.run, capture.time.sleep = fake_run, lambda _:None
            try:
                for subject, directory in config['subjects'].items():
                    capture.main(subject, directory, config['grades'], auth, 0)
                    capture.main(subject, directory, config['grades'], auth, 0)
                assert len(requests) == 4, 'saved previews must resume without network or rendering'
                try:
                    capture.main('英语', config['subjects']['数学'], config['grades'], auth, 0)
                except ValueError:
                    pass
                else:
                    raise AssertionError('cross-subject snapshot accepted')
            finally:
                capture.subprocess.run, capture.time.sleep = original_run, original_sleep
                signal.signal(signal.SIGTERM, original_signal)
                capture.STOP.clear()

            papers = load_papers(config_path)
            assert {(p['subject'],p['id']) for p in papers} == {('数学','7'),('英语','7')}
            assert all(image_path(p, 1).is_file() for p in papers)
            save(dict(raw, subjects={'数学':raw['subjects']['数学']}))
            assert [p['subject'] for p in load_papers(config_path)] == ['数学']
            save(raw)

            original_get, original_cache = matching.get_page, matching.CACHE
            matching.CACHE = root/'cache'
            def fake_page(url):
                subject = '数学' if '数学' in url else '英语'
                return (f'<h2><a href="/a/{subject}/1.html">2025北京高一{subject}</a></h2>'
                        '<a href="list_1_1.html">末页</a>')
            matching.get_page = fake_page
            try:
                for subject, directory in config['subjects'].items():
                    matching.main(subject, directory, config['grades'])
                    result = json.loads((directory/'外站目录证据.json').read_text())
                    assert result['matched_originals'] == 1
            finally:
                matching.get_page, matching.CACHE = original_get, original_cache
            for script in ('collect_catalog.py','capture_previews.py','find_same_titles.py'):
                subprocess.run([sys.executable, str(ROOT/'北京题库'/script), '--config', str(config_path), '--check'], check=True)
        finally:
            os.chdir(previous)
    print('PASS YAML validation and relative paths; shared math/English collection, capture, resume, matching and downstream paper loading')


if __name__ == '__main__':
    check()
