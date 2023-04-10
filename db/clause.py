import pathlib
import yaml

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Mapping, 상속 클래스들을 자동으로 인지하고 매핑
class Base(DeclarativeBase):
    __abstract__ = True

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

class Data(Base):
    __tablename__ = 'data'  # data 테이블과 매핑된다.
    ticker = mapped_column(String, nullable=False)
    date = mapped_column(DateTime, nullable=False)
    product = mapped_column(String, nullable=False)
    sub_title = mapped_column(String, nullable=False)
    content = mapped_column(String, nullable=False)
    doc_no = mapped_column(String, nullable=False)
    row_no = mapped_column(Integer, nullable=False, primary_key=True)
    
    def __init__(self, ticker, date, product, sub_title, content, doc_no):
        self.ticker = ticker
        self.date = date
        self.product = product
        self.sub_title = sub_title
        self.content = content
        self.doc_no = doc_no
    #    self.row_no = row_no
    #   row_no is annotated due to SQL server side auto-increment setting, so don't un-annotate.
    
   # def __repr__(self):
   #     return 'user_id : %s, user_name : %s, profile_url : %s' % (self.user_id, self.user_name, self.profile_url)

class DBConnect:
    def __init__(self) -> None:
        db_config_path = pathlib.Path(__file__).parent.joinpath('db.yaml')
        with open(db_config_path, "r") as f:
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
