import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table
from sqlalchemy.orm import sessionmaker


from clause import DBConnect, Data, Summary, Reward


# 전체 테이블 조회 (학습용)
def select_all(db_connect):
    '''
    ans = db_connect.session.query('data').all()
    df = pd.DataFrame(ans)
    '''
    sql = text('select * from data')
    result = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    return df

# 일정 수의 데이터 조회 (학습용)
def select_n(db_connect,n):
    '''
    ans = db_connect.session.query('text').limit(n).all()
    df = pd.DataFrame(ans)
    '''
    sql = text(f'select * from data limit{n}')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))
    return df

# 요약이 되지 않은 데이터 조회 (한 건 씩 출력)
def summary_unlabeled(db_connect):
    '''
    #ans = db_connect.session.query(Data).filter(Data.row_no == db_connect.session.query(Summary.row_no).filter(Summary.summary == None).first()).first()
    ans = db_connect.session.query(Data.row_no).filter(Data.row_no == db_connect.session.query(Summary.row_no).filter(Summary.summary == None).first()).first()
    df = pd.DataFrame(list(ans))
    '''
    sql = text('select * from data where data.row_no = (select summary.row_no from summary where summary.summary is null limit 1)')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))

    return df

# 생성된 요약문을 DB에 저장
def save_summary(row_no, summary):
    '''
    sum_list = pd.values.tolist()
    DBConnect.SummaryAdd(sum_list)
    '''
    db_connect.session.SummaryAdd(row_no, summary)
    db_connect.session.commit()

def save_reward_label(row_no, reward):
    '''
    reward_list = pd.values.tolist()
    DBConnect.RewardAdd(reward_list)
    '''
    db_connect.session.RewardAdd(row_no, reward)
    db_connect.session.commit()

# Reward 라벨링이 되지 않은 데이터 조회 (한 건 씩 출력)
def reward_unlabeled(db_connect):
    '''
    ans = db_connect.session.query(Data).join(Summary, Data.row_no == Summary.row_no).outerjoin(Reward, Data.row_no == Reward.row_no).filter(Reward.reward != None).all()
    df = pd.DataFrame(ans)
    '''
    sql = text('select * from data join summary on data.row_no = summary.row_no inner join reward.row_no on summary.row_no = reward.row_no where data.row_no = (select reward.row_no from reward where reward.reward is null limit 1)')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))

    return df
