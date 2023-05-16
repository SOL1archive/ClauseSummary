import re
import numpy as np
import pandas as pd

def summary_preprocessing_func1(text:str) -> str: #숫자. 형태로 되어있는것에 개행문자를 추가.
    pattern = r'(\d+)\.'
    return re.sub(pattern,r'\n\1', str(text))

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['summary'][0]
    result = summary_preprocessing_func1(text)

    print(text, result, sep='\n' + '-'*100 + '\n')
    