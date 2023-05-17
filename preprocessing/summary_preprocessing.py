import re
import numpy as np
import pandas as pd

def summary_preprocessing_func1(text:str) -> str: #숫자. 형태로 되어있는것에 개행문자를 추가.
    pattern = r'(\d+)\.'
    return re.sub(pattern,r'\n\1.', str(text))

def summary_preprocessing_func2(text:str): # (1) (2) 형태를 ,으로
    items = re.split(r'\(\d+\)', text)
    if len(items) > 1:
        items[1] = items[0]+items[1]
        del items[0]
    return ','.join(items)

def summary_preprocessing_func(text: str):
    text = summary_preprocessing_func1(text)
    text = summary_preprocessing_func2(text)
    
    return text

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['summary'][0]
    result = summary_preprocessing_func1(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
    
