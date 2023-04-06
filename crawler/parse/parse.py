import os
import pathlib
import ftplib

import unit_split
import utils
from db.clause import *

# TODO: FTP, Local 파일 처리 일반화

if __name__ == '__main__':
    # DBConnect 객체 생성
    connect = DBConnect()
    yaml_path = pathlib.Path(__file__).parent.joinpath('parser-config.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    if config['Local']:
        # 확인해야하는 디랙토리 리스트를 불러옴
        csv_path = pathlib.Path(__file__).parent.joinpath('target_dirs.csv')
    else:
        # FTP 접속
        ftp = ftplib.FTP(config['FTP']['host'], config['FTP']['user'], config['FTP']['password'])
        ftp.cwd(config['FTP']['dir'])

    with open(csv_path, 'r') as f:
            target_dirs = [pathlib.Path(str_path) for str_path in f.readlines()]

    # 각 파일에 대해 처리함
    for dir in target_dirs:
        os.chdir(dir)

        for file in os.listdir(dir):
            if utils.is_pdf(file):
                pages = unit_split.get_pdf_pages(file)
                page_units = unit_split.split_to_unit(pages)

                # TODO: Erase Index Pages

                file_name_split = file[:-4].split('-')
                ticker = file_name_split[0]
                product = file_name_split[1]
                doc_no = int(file_name_split[2])

                # TODO date, sub_title, contents 추가
                

                connect.add()

    if not config['Local']:
        # FTP 접속 종료
        ftp.quit()

    connect.commit()
