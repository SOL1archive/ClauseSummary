import sqlalchemy
from sqlalchemy import create_engine, Column
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine()

Base = declarative_base()

class Clause(Base):
    __tablename__ = ''
    
