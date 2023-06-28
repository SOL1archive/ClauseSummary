import os

# import PyPDF2
import pdfplumber
import pandas as pd
from utils import *

# pdf의 모든 페이지를 불러옴
def get_pdf_pages(file):
    EOF_MARKER = b'%%EOF'
    
    '''
    with open(file, 'rb') as f:
        contents = f.read()

    if EOF_MARKER in contents:
        contents = contents.replace(EOF_MARKER, b'')
        contents = contents + EOF_MARKER
    else:
        contents = contents[:-6] + EOF_MARKER

    with open(file.replace('.pdf', '') + '_fixed.pdf', 'wb') as f:
        f.write(contents)
    '''

    with open(file, 'ab') as f:
        f.write(EOF_MARKER)

    # PyPDF2
    # f = open(file, 'rb')
    # pdf_reader = PyPDF2.PdfReader(f)
    # full_doc = [page.extract_text() for page in pdf_reader.pages]
    # f.close()
  
    # pdfplumber
    with pdfplumber.open(file) as pdf:
        full_doc = [page.extract_text() for page in pdf.pages]
    return full_doc

# 목차 텍스트를 기준으로 데이터를 불러옴
# 확인해야 할 점: 단순히 '목차'라는 단어가 나오는 경우에도 분리가 될 수 있음
def split_to_unit(full_doc: list):
    # 목차를 지정하는 키워드 저장
    with open('C:\\Users\\jeff4\Downloads\\ClauseSummary-main\\parser-v0\\index.csv', 'r', encoding='utf-8') as f:
        index_name_lt = []
        for line in f.readlines():
            index_name_lt.append(line.strip('\n'))            ### UPDATE

    # 목차가 나오는 페이지를 찾음
    index_pages = []
    for i, page in enumerate(full_doc):
        for index_name in index_name_lt:
            # 목차가 나오는 페이지를 찾음
            #if isin(page, index_name) or False: # TODO: 목차가 나오는 페이지를 찾는 방법을 개선해야 함
            if index_name in page:
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
