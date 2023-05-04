# Crawler
- [x] FTP 서버 연결 구현
- [ ] 파일 계층 리스트 기입
- [x] 테스트 코드 작성
- [ ] 테스트

# DB
- [ ] Human Feedback / Training을 위한 `sqlalchemy` 쿼리문 작성
    - [Tips](https://soogoonsoogoonpythonists.github.io/sqlalchemy-for-pythonist/tutorial/)
- [ ] Human Feedback & Summarization DB 생성

# Data & RLHF
- [ ] `chat_gpt` 혹은 `clova_api`를 이용한 요약 코드 작성(쿼리문 작성 선행)
- [ ] RLHF를 위한 가벼운 챗봇 기반 인터페이스 개발
    - [ ] 챗봇 플랫폼 결정
    - [ ] API 문서 여부 확인
    - [ ] 입출력 구현
    - [ ] DB 구현

# Training
- [ ] Main Model(Summarization Model) 학습 코드 작성
- [ ] Evaluation Model 학습 코드 작성
- [ ] Reinforcement Learning 코드 작성

# 논의사항
- 법 조항 데이터 사용 여부
    - Pros
        - 유사한 데이터 형식
    - Cons
        - 실제론 다소 다를 수 있음
- 요약문 추가 Dataset \
    https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582
- 모델
    - https://huggingface.co/lcw99/t5-large-korean-text-summary
    - https://huggingface.co/eenzeenee/t5-small-korean-summarization
    - https://huggingface.co/eenzeenee/t5-base-korean-summarization
    - https://huggingface.co/gogamza/kobart-summarization
