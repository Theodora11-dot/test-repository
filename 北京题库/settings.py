"""Load the shared subject and source-directory configuration."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/config.yaml'


def load_config(path=None):
    path = Path(path if path is not None else DEFAULT_CONFIG).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {'subjects', 'grades'}:
        raise ValueError('配置必须且只能包含 subjects 和 grades')
    subjects, grades = config['subjects'], config['grades']
    if not isinstance(subjects, dict) or not subjects or any(s not in ('数学', '英语') for s in subjects):
        raise ValueError('subjects 必须是数学或英语到数据目录的非空映射')
    if (not isinstance(grades, list) or not grades or
            any(g not in ('高一', '高二', '高三') for g in grades) or len(set(grades)) != len(grades)):
        raise ValueError('grades 必须是高一、高二、高三组成的非空列表，不能重复')
    directories = {}
    for subject, value in subjects.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('学科数据目录不能为空')
        directory = (path.parent / Path(value).expanduser()).resolve()
        if directory == ROOT or any(directory == p or p in directory.parents for p in
                                    (ROOT/'北京题库', ROOT/'config', ROOT/'题库处理')):
            raise ValueError('原始数据目录不能设在代码、配置或处理结果目录中')
        if directory in directories.values():
            raise ValueError('不同学科必须使用不同数据目录，避免清单互相覆盖')
        directories[subject] = directory
    return dict(subjects=directories, grades=grades)
