import ts_dataset_parser as ps
import pandas as pd
import pathlib
import yaml

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Mapping
class Base(DeclarativeBase):
    __abstract__ = True

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

class Data(Base):
    __tablename__ = 'dataset'  #updating dataset table 
    text = mapped_column(String, nullable=False)
    title = mapped_column(String, nullable=False)
    id = mapped_column(Integer, nullable=False, primary_key=True)
    
    def __init__(self, text, title, id):
        self.text = text
        self.title = title
        self.id = id

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
