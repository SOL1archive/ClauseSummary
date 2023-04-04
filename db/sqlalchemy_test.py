#pip install pymysql
#pip install sqlalchemy
#required

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import pymysql
import yaml

# DB 연결
from sqlalchemy import create_engine
with open("db.yaml", "r") as f:
    credentials = yaml.safe_load(f)
db_url = f"mysql+pymysql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
engine = create_engine(db_url)


# engine 종속적 session 정의
Session = sessionmaker(engine)
session = Session()

# Mapping, 상속 클래스들을 자동으로 인지하고 매핑
Base = declarative_base()

# 테이블 생성


class Data(Base):
    __tablename__ = 'data'  # data 테이블과 매핑된다.
    ticker = Column(Integer, nullable=False)
    date = Column(String, nullable=False)
    product = Column(String, nullable=False)
    sub_title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    doc_no = Column(Integer, nullable=False)
    row_no = Column(Integer, nullable=False, primary_key=True)
    
    def __init__(self, ticker, date, product, sub_title, content, doc_no):
        self.ticker = ticker
        self.date = date
        self.product = product
        self.sub_title = sub_title
        self.content = content
        self.doc_no = doc_no
    #    self.row_no = row_no
    
   # def __repr__(self):
   #     return 'user_id : %s, user_name : %s, profile_url : %s' % (self.user_id, self.user_name, self.profile_url)

    
test = Data(123456, "2023-04-04 20:00:00", 'hj', 'test', 'for test 2', 1)

session.add(test)
session.commit()
