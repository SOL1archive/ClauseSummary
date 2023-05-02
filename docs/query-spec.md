# DB Specifications
## Label Data Generation
- Select: 불러올 데이터

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`product`||
    |`sub_title`||
    |`content`||
    |`row_no`||

- Insert: 저장할 데이터

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

## Human Feedback Query
- Select: 불러올 데이터

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`product`||
    |`sub_title`||
    |`content`||
    |`row_no`||

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

    `row_no`로 Join 해서 불러오기

- Insert: 저장할 데이터

    - Table: Feedback

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`score`||

    `row_no`로 인덱싱해서 `score` 저장

## Training
- Select: 불러올 데이터

    - Table: Data

    |Column|Desc.|
    |-|-|
    |`product`||
    |`sub_title`||
    |`content`||
    |`row_no`||

    - Table: Summary

    |Column|Desc.|
    |-|-|
    |`row_no`||
    |`summary`||

    `row_no`로 Join 해서 불러오기
