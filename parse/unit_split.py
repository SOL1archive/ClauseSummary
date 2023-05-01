import PyPDF2
import pandas as pd
from utils import *

# pdf의 모든 페이지를 불러옴
def get_pdf_pages(file):
    with open(file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)

    return [page.extract_text() for page in pdf_reader.pages]

# 목차 텍스트를 기준으로 데이터를 불러옴
# 확인해야 할 점: 단순히 '목차'라는 단어가 나오는 경우에도 분리가 될 수 있음
def split_to_unit(full_doc: list):
    # 목차를 지정하는 키워드 저장
    with open('./crawler/parse/index.csv', 'r') as f:
        index_name_lt = []
        for line in f.readlines():
            index_name_lt.append(line[:-1])
    
    # 목차가 나오는 페이지를 찾음
    index_pages = []
    for i, page in enumerate(full_doc):
        for index_name in index_name_lt:
            # 목차가 나오는 페이지를 찾음
            if isin(page, index_name) or False: # TODO: 목차가 나오는 페이지를 찾는 방법을 개선해야 함
                index_pages.append(i)
                break

    # 목차가 나오는 페이지를 기준으로 문서를 분리함
    index_pages.append(len(full_doc))
    result_doc = []
    for start, end in zip(index_pages, index_pages[1:]):
        result_doc.append(
            full_doc[start:end]
        )

    return result_doc
