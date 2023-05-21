import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
print(sys.path)

from db.clause import *

if __name__ == '__main__':
    # DBConnect 객체 생성
    connect = DBConnect()

    for i in range(1,11):
        connect.add(str(i),\
                f'2021-01-{i:02}',\
                f'product{i}',\
                f'sub_title{i}',\
                f'content{i}',\
                i)

    connect.commit()
