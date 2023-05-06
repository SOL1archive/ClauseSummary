import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table
from sqlalchemy.orm import sessionmaker, 

from clause import Data, DBConnect

# 전체 테이블 조회 (학습용)
def select_all() -> pd.DataFrame:
    ans = DBconnect.session.query(data).all()
    return ans

# 일정 수의 데이터 조회 (학습용)
def select_n(n) -> pd.DataFrame:
    ans = DBconnect.session.query(data).limit(5).all()
    return ans

# 요약이 되지 않은 데이터 조회 (한 건 씩 출력)
def summary_unlabeled() -> pd.DataFrame:
    ans = DBconnect.session.query(data).all()
    ans = ans.filter(data.id.not_in(DBconnect.session.query(data.id).all()))
    return ans

# 생성된 요약문을 DB에 저장
def save_summary(row_no, summary) -> None:
    #add 함수 이용해야 할 듯?
    
    
# Reward 라벨링이 되지 않은 데이터 조회 (한 건 씩 출력)
def reward_unlabeled() -> pd.DataFrame:
    ans = DBconnect.session.query(sum_data).filter_by(id=i).all()
    return ans
