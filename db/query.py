import numpy as np
import pandas as pd
from sqlalchemy import select, Connection, MetaData, Table
from sqlalchemy.orm import sessionmaker

from clause import Data, DBConnect

# 전체 테이블 조회
def select_all() -> pd.DataFrame:
    pass

# 특정 product 조회
def product(product) -> pd.DataFrame:
    pass
