import PyPDF2
from utils import *

def get_pdf_pages(file):
    with open(file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)

    return [page.extract_text() for page in pdf_reader.pages]

def split_to_unit(full_doc: list):
    with open('./crawler/parse/index.csv', 'r') as f:
        index_name_lt = []
        for line in f.readlines():
            index_name_lt.append(line[:-1])
    
    index_pages = []
    for i, page in enumerate(full_doc):
        for index_name in index_name_lt:
            if isin(page, index_name):
                index_pages.append(i)
    
    result_doc = []
    for start, end in zip(index_pages, index_pages[1:]):
        result_doc.append(
            full_doc[start:end]
        )

    return result_doc
