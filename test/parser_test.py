import os
import sys
import pytest

from parse import parse_clauses, main

PRTNT_TEST = True

file_list = [

]

@pytest.mark.parametrize('file', file_list)
def single_file_parse_test(file):
    ticker, date, product, sub_title, contents, doc_no = parse_clauses(file)
    if PRTNT_TEST:
        print(ticker, date, product, sub_title, contents, doc_no, sep='\n\n')
    assert type(ticker) == str
    assert contents.count('\n') > 0
    assert contents.count('\n') < 1000
