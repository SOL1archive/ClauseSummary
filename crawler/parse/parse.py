import os
import pathlib

import unit_split
import utils
from db.clause import *

if __name__ == '__main__':
    connect = DBConnect()
    csv_path = pathlib.Path(__file__).parent.joinpath('target_dirs.csv')
    with open(csv_path, 'r') as f:
        target_dirs = [pathlib.Path(str_path) for str_path in f.readlines()]
    
    for dir in target_dirs:
        os.chdir(dir)

        for file in os.listdir(dir):
            if utils.is_pdf(file):
                pages = unit_split.get_pdf_pages(file)
                page_units = unit_split.split_to_unit(pages)

                # TODO: Erase Index Pages

            # TODO: DB Updates
            connect.add()

    connect.commit()
