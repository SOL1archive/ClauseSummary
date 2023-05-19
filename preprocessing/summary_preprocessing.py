import re
import numpy as np
import pandas as pd

def summary_preprocessing_func1(text:str) -> str: #숫자. 형태로 되어있는것에 개행문자를 추가.
    text = re.sub(r'(\d+)\.',r'\n\1.', str(text))
    return text

def summary_preprocessing_func2(text:str): # (1) (2) 형태를 ,으로
    items = re.split(r'\(\d+\)', text)
    if len(items) > 1:
        items[1] = items[0]+items[1]
        del items[0]
    return ','.join(items)

def summary_preprocessing_func3(text:str) -> str:
        text = re.sub(r"\b갑\b", r'갑\b', str(text))
        text = re.sub(r"\b을\b", r'을\b', str(text))
        text = re.sub(r"\b병\b", r'병\b', str(text))
        text = re.sub(r"\b정\b", r'정\b', str(text))
        return text

def summary_preprocessing_func(text: str):
    text = summary_preprocessing_func1(text)
    text = summary_preprocessing_func2(text)
    text = summary_preprocessing_func3(text)
    
def summary_preprocessing_func4(text:str): # 1. 2. 등을 제 1조 2 항 등으로 바꿔줌
    p_r = re.finditer(r"제 *\d조", str(text))
    paragraphs = [m.span() for m in p_r]#[:][1] # end index of each paragraph
    #print(paragraphs[0])
    if len(paragraphs) == 0:
        return text
    print("Paragraphs idex : ", paragraphs)
    paragraphs = [p[1] for p in paragraphs]
    paragraph_nums = [m.match() for m in p_r]

    a_r = re.finditer(r"\d+\.", str(text))
    articles = [m.span() for m in a_r]
    print("article length", len(articles))
    article_nums = [int(re.search(r"\d", s.match())) for s in a_r]
    #s = re.split(r"[0-9]+\.", str(text))
    paragraph_idx = 0
    for i in range(len(articles)):
        while paragraph_idx < len(paragraphs) and articles[i][0] > paragraphs[paragraph_idx]:
            paragraph_idx += 1
        paragraph_idx -= 1
        if paragraph_idx < 0:
            continue
        replaced = "제 %s조 %d항"%(paragraph_nums[paragraph_idx], article_nums[i])
        print(replaced)
        text = text[:articles[i][0]] + replaced + text[articles[i][1]:]
    return text

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['summary'][0]
    result = summary_preprocessing_func3(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
