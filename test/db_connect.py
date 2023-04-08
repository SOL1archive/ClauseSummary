from db.clause import *

if __name__ == '__main__':
    # DBConnect 객체 생성
    connect = DBConnect()
    for i in range(10):
        connect.add(i, f'2021-01-{i:02}', f'product{i}', f'sub_title{i}', f'content{i}')
    connect.commit()
