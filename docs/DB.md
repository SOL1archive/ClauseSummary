# Data: ORM

|Column|Type|Description|Note|
|-|-|-|-|
|ticker|Integer|||
|date|String|||
|product|String|||
|sub_title|String|||
|content|String|||
|doc_no|Integer|||
|row_no|Integer||Primary Key|

# DBConnect
SQLAlchemy의 세부사항들을 추상화하여 간단하게 한 API. `db.yaml`에 설정 사항들을 저장함.

- `db.yaml` Requirments
    - `host`: DB IP
    - `port`: DB 연결 포트
    - `user`: DB 사용자
    - `password`: DB 비밀번호
    - `database`: DB 지정

### Methods
- `add(*argv, **kwarg)`:\
    데이터를 입력받아 캐싱함.
- `commit()`:\
    저장한 데이터를 DB에 전송함.
