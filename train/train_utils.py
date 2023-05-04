import pandas as pd
from datasets import Dataset, DatasetDict

# DataFrame을 Huggingface Dataset으로 변환. 원본 DataFrame은 자동으로 삭제되니 주의
def df2huggingface_dataset(df):
    dataset = Dataset.from_pandas(df)
    del df
    return dataset
