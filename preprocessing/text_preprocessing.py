import re
import pandas as pd

def text_preprocessing_func(text):
    return re.sub(r'\n[\n ]+', '\n', text)

# ________________________________  새로 추가됨 ____________________________________
# 목차 제거
def find_table_of_contents(text: str):
    lines = text.split("\n")
    r = ""
    for line in lines:
        comma_matches = re.findall(r"·{5,}", line)
        if len(comma_matches) == 0 and "목 차" not in line and "목차" not in line:
            r = r+line+"\n"
    return r
    
def remove_duplicate_chars(text:str, threshold = 5, verbose = False): 
    exception_lst = (" ", "\n", "·", ".", ",")
    never_duplicate_lst = ("(", ")")
    result = ""
    for line in text.split("\n"):
        if len(line) > 0:
            r = line[0]
            duplicate_flag = False
            for i in range(1, len(line)):
                if line[i-1] != line[i] or line[i] in exception_lst:
                    r = r + line[i]
                elif line[i] in never_duplicate_lst:
                    duplicate_flag = True
            if len(line) - len(r) <= threshold and not duplicate_flag:
                result = result + line + "\n"
            else:
                duplicate_flag = True
                result = result + r + "\n"
                if verbose:
                    print("중복된 단어들을 삭제합니다.\n")
    return result


if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['text'][0]
    result = text_preprocessing_func(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
    
