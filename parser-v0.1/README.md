1. unit_split.py -> [목차] 1D array
  pdfplumber로 pdf를 열고, 목차가 나오는 페이지를 기준으로 문서를 구분함. 
  
2. clause_split.py -> [목차][조항] 2D array
  목차를 기준으로 구분된 1D array에서 목차에 해당하는 페이지를 모두 지우고, [목차][조항]으로 구성된 array를 생성
 
3. debugger.py
  unit_split.py와 clause_split.py의 함수가 valid 한지 검증하기 위한 main 함수
  함수의 argument에 파일명을 전달할 때 "./path/파일명"으로 전달해야 함.
  
4. parser_test.py
  파서에 대한 테스트 코드
5. parse.py
  파서 컨트롤러. 로컬에 대한 파싱과 FTP 서버에 대한 파싱을 기본으로 제공함.
