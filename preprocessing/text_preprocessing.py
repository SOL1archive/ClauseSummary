import re
import pandas as pd

def text_preprocessing_func(text):
    return re.sub(r'\n[\n ]+', '\n', text)

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['text'][0]
    result = text_preprocessing_func(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
    