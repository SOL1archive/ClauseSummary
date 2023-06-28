import sys
import os
import datetime
import pathlib
import ftplib
import yaml

import unit_split
import strip
import clause_split as cs
import utils
import db.clause as clause

# 문서 하나를 파싱함
def parse_clauses(file):
    pages = unit_split.get_pdf_pages(file)
    page_units = unit_split.split_to_unit(pages)
    clauses = cs.articles2clauses(page_units)
    
    file_name_split = os.path.basename(file)[:-4].split('-')
    label = file_name_split[0]
    title = file_name_split[1]
    id = int(file_name_split[2])


    for i, clause in enumerate(clauses):
        sub_title = '일반조항' if i == 0 else f'부칙{i}'
        text = '\n'.join(clause)

        yield (text, title, sub_title, id, label)

def main_local():
    connect = clause.DBConnect()
    yaml_path = pathlib.Path(__file__).parent.joinpath('parser-config.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
  
    csv_path = pathlib.Path(__file__).parent.joinpath('target_dirs.csv')
    with open(csv_path, 'r') as f:
            target_dirs = [pathlib.Path(str_path) for str_path in f.readlines()]

    # 각 파일에 대해 처리함
    for dir in target_dirs:
        for file in dir.iterdir():
            if utils.is_pdf(file):
                for (text, title, sub_title, id, label) in parse_clauses(file):
                    connect.add_data(text, title, sub_title, id, label)
    


def main_ftp():
    connect = clause.DBConnect()
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
        for file in dir.iterdir():
            if utils.is_pdf(file):
                for (text, title, sub_title, id, label) in parse_clauses(file):
                    connect.add_data(text, title, sub_title, id, label)
    
    ftp.quit()
    connect.commit()

if __name__ == '__main__':
    local_target = input('Local target? (y/n) ').lower() == 'y'
    if local_target:
        main_local()
    else:
        main_ftp()
