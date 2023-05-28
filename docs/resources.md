## Dataset
- 추가 데이터셋
    - 요약문 데이터셋(Labeled) \
        https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582
    - 약관 데이터셋(Unlabeled) \
        https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580
    - Pipeline
        https://aiopen.etri.re.kr/corpusModel
        
## Models
- 모델(Summarize): 학습 끝난 모델에 [x] 표시
    - [x] https://huggingface.co/lcw99/t5-large-korean-text-summary
    - [ ] https://huggingface.co/eenzeenee/t5-base-korean-summarization
    - [ ] https://huggingface.co/gogamza/kobart-summarization
    - [ ] https://nlp.jbnu.ac.kr/papers/leegh_hclt2021_prefixlm.pdf 
    
- 모델(Reward) https://huggingface.co/blog/rlhf
    - ** https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2?text=I+hate+you
        : Comment 충분, max sequence 알 수 없음
    - * https://github.com/lvwerra/trl/tree/main/examples/summarization
      ** https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2
        : Comment 부족
    - *** https://github.com/openai/summarize-from-feedback
        : Comment, Example 충분, 
