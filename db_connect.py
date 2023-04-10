import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
print(sys.path)

import datetime
from db.clause import *

if __name__ == '__main__':
    # DBConnect 객체 생성
    connect = DBConnect()
    for i in range(10):
        connect.add(int(i), int(datetime.datetime(f'2021-01-{i:02}')), int(f'product{i}'), int(f'sub_title{i}'), int(f'content{i}'), int(i))
    connect.commit()
