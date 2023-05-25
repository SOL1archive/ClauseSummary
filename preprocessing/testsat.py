import re
import pandas as pd

def change_to_number_comma(text:str):
    example = re.split(r'\d+\.', text)
    if len(example) > 1:
        example[1] = example[0] + example[1]
        del example[0]
    return ''.join(example)

def change_to_number_comma2(text:str):
    example1 = re.search(r'\제\s\d+\s\조')
    example = re.sub(r'\d+\.',r'제 ', text)
