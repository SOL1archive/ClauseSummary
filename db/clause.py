import datetime
import yaml

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql

# Mapping, 상속 클래스들을 자동으로 인지하고 매핑
Base = declarative_base()

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

class DBConnect:
    def __init__(self) -> None:
        with open("db.yaml", "r") as f:
            credentials = yaml.safe_load(f)
        self.db_url = f"mysql+pymysql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
        self.engine = create_engine(self.db_url)

        # engine 종속적 session 정의
        self.Session = sessionmaker(self.engine)
        self.session = self.Session()
    
    def add(self, *argv, **kwarg):
        row = Data(*argv, **kwarg)
        self.session.add(row)

    def commit(self):
        self.session.commit()
