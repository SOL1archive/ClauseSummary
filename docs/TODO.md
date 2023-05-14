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
- 추가 데이터셋
    - 요약문 데이터셋(Labeled) \
        https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582
    - 약관 데이터셋(Unlabeled) \
        https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580
    - Pipeline
        https://aiopen.etri.re.kr/corpusModel
        
- 모델(Summarize)
    - https://huggingface.co/lcw99/t5-large-korean-text-summary
    - https://huggingface.co/eenzeenee/t5-small-korean-summarization
    - https://huggingface.co/eenzeenee/t5-base-korean-summarization
    - https://huggingface.co/gogamza/kobart-summarization
    
    - https://nlp.jbnu.ac.kr/papers/leegh_hclt2021_prefixlm.pdf 
    - https://github.com/HaloKim/KorBertSum
    - https://github.com/raqoon886/KorBertSum
    - https://velog.io/@raqoon886/KorBertSum-SummaryBot (ChatBot)
    
- 모델(Reward) https://huggingface.co/blog/rlhf
    - ** https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2?text=I+hate+you
        : Comment 충분, max sequence 알 수 없음
    - * https://github.com/lvwerra/trl/tree/main/examples/summarization
      ** https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2
        : Comment 부족
    - *** https://github.com/openai/summarize-from-feedback
        : Comment, Example 충분, 
