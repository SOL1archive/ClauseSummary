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

<<<<<<< Updated upstream
def summary_preprocessing_func(text: str):
    text = summary_preprocessing_func1(text)
    text = summary_preprocessing_func2(text)
    
    return text
=======
def summary_preprocessing_func2(text:str) -> str:
    pattern = r'\b?갑?\b|\b?을?\b|\?b병?\b|\b?정?\b'
    if pattern == "\b갑\b":
        return re.sub(pattern, r'갑\b', str)
    elif pattern == "\b을\b":
        return re.sub(pattern, r'을\b', str)
    elif pattern == "\b병\b":
        return re.sub(pattern, r'정\b', str)
    elif pattern == "\b병\b":
        return re.sub(pattern, r'정\b', str)
>>>>>>> Stashed changes

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['summary'][0]
    result = summary_preprocessing_func2(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
    
