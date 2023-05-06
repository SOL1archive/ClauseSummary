import sys
import os
import datetime
import pathlib
import ftplib
# import yaml
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))

import unit_split
import strip
import clause_division as cd
import utils
#from db.clause import *

# 문서 하나를 파싱함
def parse_clauses(file):
    pages = unit_split.get_pdf_pages(file)
    page_units = unit_split.split_to_unit(pages)
    clauses = cd.articles2clauses(page_units)
    
    file_name_split = file[:-4].split('-')
    ticker = file_name_split[0]
    product = file_name_split[1]
    doc_no = int(file_name_split[2])
    for i, clause in enumerate(clauses):
        sub_title = '일반조항' if i == 0 else f'부칙{i}'
        contents = '\n'.join(clause)
        date = datetime.datetime.now().strftime('%Y-%m-%d')

        yield (ticker, date, product, sub_title, contents, doc_no)

def main():
    connect = DBConnect()
    yaml_path = pathlib.Path(__file__).parent.joinpath('parser-config.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    csv_path = pathlib.Path(__file__).parent.joinpath('target_dirs.csv')
    with open(csv_path, 'r') as f:
            target_dirs = [pathlib.Path(str_path) for str_path in f.readlines()]
    
    # FTP 접속
    ftp = ftplib.FTP(config['host'], config['user'], config['password'])

    # 각 파일에 대해 처리함
    for dir in target_dirs:
        ftp.cwd(dir)

        for file in ftp.nlst():
            if utils.is_pdf(file):
                for (ticker, date, product, sub_title, contents, doc_no) in parse_clauses(file):
                    connect.add(ticker, date, product, sub_title, contents, doc_no)
    
    ftp.quit()
    connect.commit()

if __name__ == '__main__':
    main()
