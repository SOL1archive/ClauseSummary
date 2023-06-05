# ClauseSummary
> Clause Summarization Model

> 2023-1 YCC Project

> website repo https://github.com/HeIIowrld/ToSAN

**[TODO](./docs/TODO.md)**

## Introduction
### 프로젝트 목적
금융상품의 종류와 특성은 다양하다. 주식, 채권부터 콜 옵션, 풋 옵션, CDS, CDO, ETF까지 다양한 기초상품과 파생상품이 존재한다. 이러한 금융상품들은 큰 범주에서는 분류가 되지만 세부적인 조항, 기초상품 등에 따라 그 특성과 리스크가 달라지기도 하며, 심지어 이전에는 없었던 새로운 종류의 금융상품이 판매되기도 한다. 더 나아가 각 나라의 금융상품들은 각 나라의 금융  법, 정책, 규제에 강한 영향을 받기 때문에 해당 국가의 금융 시장 특성과 금융 법, 정책은 국가간 금융 투자에 대한 장벽으로 작용했다.

이는 여러 문제들을 낳았는데, 대표적인 문제로는 경제적 비효율성과 투자자가 의도치 않는 위험을 떠안는 것을 들 수 있다. 시장 투자에 대한 진입 장벽은 시장 참여자의 수를 줄이고, 가격 변동에 대한 경직성을 만듦과 동시에 시장 전반에 자금을 공급하는 금융시장 내에서 최대로 운용될 수 있는 자금보다 더 적은 자금이 운용되게 한다. 더 적은 자금은 시장 내 자본의 감소로 이어지므로, 투자가 최대 효용 수준으로 이루어지지 않아 경제적 비효율성이 발생한다.

또 다른 문제는 투자자가 금융상품의 특성을 파악하지 못하는 것이 있다. 금융상품의 특성을 이해하지 못하는 것은 투자자가 투자 의도와는 다른 특성을 가진 금융상품을 구매하여 의도치 않은 위험을 떠안을 수 있게 한다. 이는 경제, 금융에 대한 이해가 부족한 투자자들이 투기성 금융상품을 충동적으로 구매하거나, 금융 상품 구매 자체를 꺼리게 하는 유인이 된다.

이러한 이유로 인해 정보화, 세계화 시대에도 불구하고, 금융 시장 참여자의 수와 자본은 효율적이지 못하며, 국가간 금융 투자는 대형 투자기관을 중심으로만 이루어졌다. 본 프로젝트는 대규모 언어 모델(LLM)을 사용해 자연어 처리 모델이 국내 혹은 해외의 금융상품 약관을 대신 이해하고 요약 정보를 제공해 경제에 대한 지식이 부족하거나 국내 및 해외의 금융 법, 정책에 대한 이해가 부족한 투자자에게 정보를 이해하기 쉽도록 가공하도록 하고자 한다. 개별 투자자의 합리적인 금융 투자를 하도록 도움으로써, 금융 시장의 건전성과 경제 민주화를 달성하는 데 기여하고자 한다.

### 계획
- 1주차: 전체 일정 세부 조정 및 시간 조율
  - 의견을 추가적으로 수합해 계획의 보완, 개선 방안 논의
  - 정기적 모임 시간과 세부 일정 조율
- 2주차: 웹 크롤러 설계 및 개발 방향 논의
  - 웹 크롤러와 파서, DB 설계와 개발 역할 할당
- 3~4주차: 웹 크롤러 개발
  - 개발 및 주기적 코드리뷰 진행
- 5주차: 크롤링 테스트 및 크롤링 (진행중)
  - 크롤링 테스트를 통해 크롤러 테스트
  - NAVER 클라우드 서버를 이용해 크롤링 수행 및 DB 구축
- 6주차: EDA 및 필요 시 추가 데이터 수집
  - 데이터 탐색을 통해 데이터의 대략적인 특성을 파악하고 필요 시 추가로 데이터 수집
- 7주차: Pre-trained 모델 성능 테스트 및 선택
  - 여러 Pre-trained 모델 중 좋은 성능을 보이는 모델을 선택
- 8~9주차: 모델 설계/구현 및 학습 환경 구축
- 10~11주차: 모델 학습 및 검증, 튜닝
  - 모델의 학습 상황과 성능을 고려해 하이퍼파라미터를 튜닝
  - 모델을 원하는 task에 맞추어 fine-tuning 진행
  - NAVER 클라우드 GPU 서버 혹은 로컬환경에서 모델 학습 진행
- 12주차: 모델 테스트 및 Metric 측정
- 13주차: 베타 테스트 및 피드백, 추후 발전 방향 모색
  - 다양한 지식 배경을 가진 사람들에게 베타 테스트를 제공하고 피드백을 받아 모델의 개선 방향을 모색.

### 최종 목표 및 기대효과
본 프로젝트의 최종 목표를 두 가지로 나눌 수 있다. 첫째, 개별 투자자들이 직면한 여러 금융 시장 진입 장벽을 없애는 것이다. 둘째, 금융상품과 경제에 대한 이해가 적은 개별 투자자들에게 금융 시장과 금융 상품에 대한 이해를 도움으로써 합리적인 금융 투자와 금융 시장의 건전성을 획득하는 것이다. 최근 자연어 처리 모델들이 발달하며, 많은 연구가 진행된 언어의 경우, 사람이 작성한 글과 기계가 생성한 텍스트를 구분하기 어려울 정도로 발전한 모습을 보인다. 법규에 맞추어 작성된 금융상품 설명서의 변칙성이 적은 특성과, 국립국어원, 공공 데이터 포털 등 다양한 기관에서 제공하는 자연어 데이터(Corpus)를 이용하여 모델을 개발하면, 이해하기 쉬운 언어로 풀어서 실제 사용자들이 약관 내용을 이해하는데 도움을 줄 수 있을 것이라 기대한다. 그리고 이에 대한 성과는 모델의 사용자들이 얻은 효용과 만족도를 통해 확인하고 피드백을 얻을 수 있을 것이다.
