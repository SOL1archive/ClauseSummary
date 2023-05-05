# `dataset`
## `ts_dataset_parser.py`
[약관 텍스트 분석 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580) 파싱

- `load_dataset_from_dir(path)` \
    HuggingFace Dataset으로 데이터셋을 파싱함.

- `parse_dir(path)` \
    전체 디렉토리를 파싱함.

- `parse_single_labeled_dir(path)` \
    하나의 라벨링된 디렉토리를 파싱함

- `parse_file(file)` \
    하나의 파일 내에서 조항을 파싱함
