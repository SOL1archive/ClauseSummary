# DB Specifications
> 구현해야할 함수 이름은 `:` 이후에 있음

> 구현은 [db/query.py](../db/query.py) 에 구현.

## Label Data Generation
- Select: 불러올 데이터(한 건씩): `summary_unlabeled() -> pd.DataFrame`

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`content`||
    |`row_no`||

- Insert: 저장할 데이터: `save_summary(row_no, summary)`

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

## Human Feedback Query
- Select: 불러올 데이터(한 건씩): `reward_unlabeled() -> pd.DataFrame`

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`content`||
    |`row_no`||

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

    `row_no`로 Join 해서 불러오기

- Insert: 저장할 데이터: `save_reward_label(row_no, reward)`

    - Table: Feedback

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`score`||

    `row_no`로 인덱싱해서 `score` 저장

## Training
- Select: 불러올 데이터: `select_all() -> pd.DataFrame`, `select_n(n) -> pd.DataFrame`

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`content`||
    |`row_no`||

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

    `row_no`로 Join 해서 불러오기
