import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table
from sqlalchemy.orm import sessionmaker

from clause import Data, DBConnect

# 전체 테이블 조회 (학습용)
def select_all() -> pd.DataFrame:
    pass

# 일정 수의 데이터 조회 (학습용)
def select_n(n) -> pd.DataFrame:
    pass

# 요약이 되지 않은 데이터 조회 (한 건 씩 출력)
def summary_unlabeled() -> pd.DataFrame:
    pass

# 생성된 요약문을 DB에 저장
def save_summary(row_no, summary) -> None:
    pass

# Reward 라벨링이 되지 않은 데이터 조회 (한 건 씩 출력)
def reward_unlabeled() -> pd.DataFrame:
    pass

# Human Feedback Reward를 DB에 저장
def save_reward_label(row_no, reward) -> None:
    pass
