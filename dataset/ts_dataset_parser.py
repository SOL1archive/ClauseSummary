import sys
import os
import pathlib

from datasets import Dataset
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# HuggingFace Dataset을 이용하여 데이터셋을 로드함
def load_dataset_from_dir(path):
    df = parse_dir(path)
    return Dataset.from_pandas(df)

# 전체 디렉토리를 파싱함
def parse_dir(path):
    pd_list = []
    for path.child in path.iterdir():
        if path.child.is_dir():
            pd_list.append(parse_single_labeled_dir(path.child))

    return pd.concat(pd_list, axis=0)

# 하나의 라벨링된 디렉토리를 파싱함
def parse_single_labeled_dir(path):
    if type(path) != pathlib.Path:
        path = pathlib.Path(path)
    
    if not path.exists() or not path.is_dir():
        raise ValueError(f'{path} is not a valid directory')
    
    data = []
    label = path.name
    for child in tqdm(path.iterdir(), desc=f'Parsing {label}'):
        if child.is_file():
            row = dict()
            row['text'] = parse_file(child)
            row['title'] = child.name[child.name.find('_') + 1:child.name.rfind('_')]
            row['id'] = child.name[:child.name.find('_')]
            data.append(row)

    data['label'] = [label] * len(path.iterdir())

    return pd.DataFrame(data)

# 하나의 파일 내에서 조항을 파싱함
def parse_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
        text = (soup.select('cn').text
                .replace('<![CDATA[', '')
                .replace(']]>', '')
                )
    
    return text
