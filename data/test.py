import os
import pathlib
import sys
import csv
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import query
import clause

db_connect = clause.DBConnect()

#df.to_sql(name='labeled_pretrained', con=db_connection, if_exists='append')

a = query.summary_unlabeled(db_connect)

print(a)
