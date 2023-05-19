import re
import numpy as np
import pandas as pd

def add_newline_before_number(text:str) -> str: #숫자. 형태로 되어있는것에 개행문자를 추가.
    text = re.sub(r'(\d+)\.',r'\n\1.', str(text))
    return text

def change_it_to_a_comma(text:str): # (1) (2) 형태를 ,으로
    items = re.split(r'\(\d+\)', text)
    if len(items) > 1:
        items[1] = items[0]+items[1]
        del items[0]
    return ','.join(items)

def remove_whitespace_after_str(text:str):
        text = re.sub(r"\b갑\b", r'갑\b', text)
        text = re.sub(r"\b을\b", r'을\b', text)
        text = re.sub(r"\b병\b", r'병\b', text)
        text = re.sub(r"\b정\b", r'정\b', text)
        return text

def change_number_point(text:str): # 1. 2. 등을 제 1조 2 항 등으로 바꿔줌
    items = re.split(r'\d+\.', text)
    if len(items) > 1:
        items[1] = items[0]+items[1]
        del items[0]
    return ''.join(items)

def summary_preprocessing_func(text: str):
    text = add_newline_before_number(text)
    text = change_it_to_a_comma(text)
    text = remove_whitespace_after_str(text)
    text = change_number_point

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['summary'][0]
    result = remove_whitespace_after_str(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
