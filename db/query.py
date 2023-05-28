import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table, text
from sqlalchemy.orm import sessionmaker


from db.clause import DBConnect, Data, Summary, Reward


# 전체 테이블 조회 (학습용)
def select_all(db_connect):
    sql = text('select * from data')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    df.columns = list(ans.keys())

    return df

# 일정 수의 데이터 조회 (학습용)
def select_n(db_connect,n):
    sql = text(f'select * from data limit{n}')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    df.columns = list(ans.keys())

    return df

# 요약이 되지 않은 데이터 조회 (한 건 씩 출력)
def summary_unlabeled(db_connect):
    sql = text('select * from data where data.row_no = (select summary.row_no from summary where summary.summary is null limit 1)')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    df.columns = list(ans.keys())

    return df

def save_reward_label(db_connect: DBConnect, row_no, reward):
    db_connect.update_reward(row_no, reward)
    db_connect.session.commit()

def reward_unlabeled(db_connect):
    sql = text('select d.row_no, d.text, s.summary, r.reward from data as d join summary as s using (row_no) inner join reward as r using (row_no) where d.row_no = (select reward.row_no from reward where reward.reward is null limit 1)')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    df.columns = list(ans.keys())

    return df

# 생성된 요약문을 DB에 저장
def save_summary(db_connect, row_no, summary):
    db_connect.update_summary(row_no, summary)
    db_connect.session.commit()
