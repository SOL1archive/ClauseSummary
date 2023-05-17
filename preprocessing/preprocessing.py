from typing import Dict
import pandas as pd

from text_preprocessing import text_preprocessing_func
from summary_preprocessing import summary_preprocessing_func

def preprocessing(row: Dict[str, str]):
    text = row['text']
    summary = row['summary']
    text = text_preprocessing_func(text)
    summary = summary_preprocessing_func(summary)

    return {'text': text, 
            'summary': summary
            }

def df_preprocessing(df: pd.DataFrame):
    text_df = df[['text', 'summary']]
    text_df = text_df.apply(preprocessing, axis=1, result_type='expand')

    df[['text', 'summary']] = text_df[['text', 'summary']]

    return df
