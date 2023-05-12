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


    sql = text('select * from data where data.row_no = (select summary.row_no from summary where summary.summary is null limit 1)')
    ans = db_connect.session.execute(sql)
    df = pd.DataFrame(list(ans))

db_connect = clause.DBConnect()

#df.to_sql(name='labeled_pretrained', con=db_connection, if_exists='append')

a = query.summary_unlabeled(db_connect)

print(a)
