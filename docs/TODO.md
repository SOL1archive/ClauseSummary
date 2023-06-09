# 회의 목록
- [x] 중간 보고서 작성
- [ ] 강화학습 공부
    - [ ] REINFORCE
    - [ ] Actor-Critic
    - [ ] PPO
    - [ ] `RL4LM`
    - [ ] `TextRL`

# Crawler
- [x] FTP 서버 연결 구현
- [x] 파일 계층 리스트 기입
- [x] 테스트 코드 작성
- [x] 테스트

# DB
- [x] Human Feedback / Training을 위한 `sqlalchemy` 쿼리문 작성
    - [Tips](https://soogoonsoogoonpythonists.github.io/sqlalchemy-for-pythonist/tutorial/)
- [x] Human Feedback & Summarization DB 생성

# Data & RLHF
- [x] `chat_gpt` 혹은 `clova_api`를 이용한 요약 코드 작성(쿼리문 작성 선행)
- [ ] RLHF를 위한 가벼운 챗봇 기반 인터페이스 개발
    - [x] 챗봇 플랫폼 결정
    - [x] API 문서 여부 확인
    - [ ] 입출력 구현
    - [x] DB 구현

# Preprocessing
- [x] `text` Preprocessing (원문에 적용) \
    `text_preprocessing_func(text: str) -> str`
    - [x] Whitespace remove(`\n\n`, `\n \n`, etc.)
- [x] `summary` Preprocessing (요약문에 적용) \
    `summary_preprocessing_func(text: str) -> str` \
    - [x] `(1)` 은 제거, `(1)` 이 아닌 `(숫자)` 는 `,` 로 대체 \
        **조금 더 보강 필요** \
        `summary_preprocessing_func1(text: str) -> str`
    - [x] `숫자.` 형태로 되어 있을 때 앞에 개행문자 추가
    - [x] `숫자.`이 문장 시작에 있을 때, 가장 가까운 조항을 찾아서 `제n조의 숫사 항에서` 으로 바꾸기
    - [x] `갑`, `을`, `병`, `정` 뒤에 공백이 있을때 제거 \
        `summary_preprocessing_func2(text: str) -> str`

# Training
- [ ] Main Model(Summarization Model) 학습
    - [x] Huggingface - PyTorch 연결부 작성
    - [x] PyTorch Base 학습 코드 작성
    - [ ] 모델 학습 및 하이퍼파라미터 튜닝
- [ ] Evaluation Model 학습 코드 작성
    - [ ] Huggingface - PyTorch 연결부 작성
    - [ ] PyTorch Base 학습 코드 작성
    - [ ] Fine-Tuning 헤드 설계 및 작성
        - Classsifier or Regressor?
    - [ ] 모델 학습 및 하이퍼파라미터 튜닝
- [ ] Reinforcement Learning 코드 작성

# Test
- [ ] BLEU, ROUGE 테스트 코드
    - [ ] BLEU
    - [ ] ROUGE

# 논의사항
`resources.md` 로 옮김.
