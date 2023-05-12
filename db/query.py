import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table
from sqlalchemy.orm import sessionmaker

from clause import DBConnect, Data, Summary, Reward

'''
- db_connect 객체 입력받기
- 문서 스펙에 맞춰서 데이터 조회
- 전체 함수들 다 완성
- 
'''

# 전체 테이블 조회 (학습용)
def select_all(db_connect):
    ans = db_connect.session.query('data').all()
    df = pd.DataFrame(ans)
    return df

# 일정 수의 데이터 조회 (학습용)
def select_n(db_connect,n):
    ans = db_connect.session.query('text').limit(n).all()
    df = pd.DataFrame(ans)
    return df

# 요약이 되지 않은 데이터 조회 (한 건 씩 출력)
def summary_unlabeled(db_connect):
    req = db_connect.session.query(Summary).filter(Summary.text is 'null').get(1)
    ans = db_connect.session.query(Data).filter(Data.id == req).all()
    df = pd.DataFrame(ans)
    return df

# 생성된 요약문을 DB에 저장
def save_summary(db_connect, row_no, summary):
    sum_list = pd.values.tolist()
    db_connect.add_summary(sum_list)
    db_connect.commit()

def save_reward_label(db_connect, row_no, reward):
    reward_list = pd.values.tolist()
    db_connect.add_reward(reward_list)
    db_connect.commit()

# Reward 라벨링이 되지 않은 데이터 조회 (한 건 씩 출력)
def reward_unlabeled(db_connect):
    req = db_connect.session.query(Reward).filter(Reward.reward is 'null').get(1)
    ans = db_connect.session.query(Data).join(Summary, Data.id == Summary.id).filter(Data.id == req).all()
    df = pd.DataFrame(ans)
    return df
