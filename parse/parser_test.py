import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
import pytest

from parse import parse_clauses, main

PRTNT_TEST = True

file_list = [
#    '001.pdf',
#    '002.pdf',
    '003.pdf',
]

#@pytest.mark.parametrize('file', file_list)
def single_file_parse_test(file):
    ticker, date, product, sub_title, contents, doc_no = parse_clauses(file)
    if PRTNT_TEST:
        print(ticker, date, product, sub_title, contents, doc_no, sep='\n\n')
    assert input(contents) == 'y'
    assert type(ticker) == str
    assert contents.count('\n') > 0
    assert contents.count('\n') < 10000

if __name__ == '__main__':
    for file in file_list:
        print(os.path.exists(file))

    for file in file_list:
        if os.path.exists(file):
            single_file_parse_test(file)
