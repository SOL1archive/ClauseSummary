import pathlib
import yaml

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import TEXT, LONGTEXT

# Mapping, 상속 클래스들을 자동으로 인지하고 매핑
class Base(DeclarativeBase):
    __abstract__ = True

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

class Data(Base):
    __tablename__ = 'data'  # summary 테이블과 매핑된다.
    row_no = mapped_column(Integer, nullable=False, primary_key=True)
    text = mapped_column(LONGTEXT, nullable=False)
    title = mapped_column(LONGTEXT, nullable=False)
    id = mapped_column(Integer, nullable=False)
    label = mapped_column(TEXT)
    
    def __init__(self, row_no, text, title, id, label):
        self.row_no = [row_no]
        self.text = [text]
        self.title = [title]
        self.id = [id]
        self.label = [label]

    
class Summary(Base):
    __tablename__ = 'summary'  # summary 테이블과 매핑된다.
    row_no = mapped_column(Integer, nullable=False, primary_key=True)
    summary = mapped_column(LONGTEXT, nullable=False)
    
    def __init__(self, row_no, summary):
        self.row_no = [row_no]
        self.summary = [summary]

class Reward(Base):
    __tablename__ = 'reward'  # reward 테이블과 매핑된다.
    row_no = mapped_column(Integer, nullable=False, primary_key=True)
    reward = mapped_column(Integer)
    
    def __init__(self, row_no, reward):
        self.row_no = [row_no]
        self.reward = [reward]
        
class DBConnect:
    def __init__(self) -> None:
        db_config_path = pathlib.Path(__file__).parent.joinpath('db.yaml')
        with open(db_config_path, "r") as f:
            credentials = yaml.safe_load(f)
        self.db_url = 'mysql+pymysql://tosan:tosan@localhost/tosan'
        self.engine = create_engine(self.db_url)

        # engine 종속적 session 정의
        self.Session = sessionmaker(self.engine)
        self.session = self.Session()
    
    def add_data(self, *argv, **kwarg):
        row = Data(*argv, **kwarg)
        self.session.add(row)
        
    def add_summary(self, *argv, **kwarg):
        row = Summary(*argv, **kwarg)
        self.session.add(row)

    def add_reward(self, *argv, **kwarg):
        row = Reward(*argv, **kwarg)
        self.session.add(row)
    
    def execute(self):
        self.session.execute()

    def commit(self):
        self.session.commit()

    def close(self):
        self.session.close()