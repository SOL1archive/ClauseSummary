import re
import pandas as pd
from parser_preprocessing import remove_number_point, remove_duplicated_chars, remove_special_characters, remove_new_line_character, remove_number_special_characters, remove_number_page

# 목차 제거
# 열쇠글 문자열이 있는 줄은 삭제한다.
def remove_table_of_contents(text: str, keywords = ("목 차", "목차", "Contents", "contents", "CONTENTS", "table of contents", "Table of Contents", "·····", ".....", "□ "), look_ahead = 5):
    lines = text.split("\n")
    r = ""
    content_lines = 0
    for line in lines:
        flag = False
        for keyword in keywords:
            if keyword in line:
               flag = True
               content_lines = look_ahead
        if not flag:
            if content_lines > 0:
                content_lines -= 1
                if find_table_of_contents_surgical(line, keywords):
                    content_lines = look_ahead
                else: r = r+line+"\n"
            else: r = r+line+"\n"
    return r
# 삭제하고 난 몇 줄 이내에 특정 패턴이 있는 문자열을 검출한다.
def find_table_of_contents_surgical(text: str, keywords, patterns = (r"제\d+조\(.+?\)",r"제\d+조\[.+?\]", r"제\d+관\(.+?\)", r"제\d+관 .+?\(.+?\)", r"제_ .+?")):
    flag = False
    for pattern in patterns:
        if len(re.findall(pattern, text)) != 0:
            flag = True
    return flag
    
def remove_duplicate_chars(text: str, threshold=5, verbose=False): 
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

def remove_shit(text: str): # 테스트 안 해봄.
    string = """(cid:1245)(cid:2052)(cid:3339) (cid:3303)(cid:2555)(cid:2504)(cid:3311) (cid:2346)(cid:2266)(cid:2628)(cid:1808)(cid:2049)(cid:3319) (cid:11)(cid:1)(cid:34)(cid:42)(cid:40)(cid:1)(cid:2263)(cid:1104) (cid:2439)(cid:1899)(cid:1851)(cid:2978) (cid:2635)(cid:3104)(cid:1394)(cid:2250)(cid:1413) (cid:1234)(cid:1819)(cid:9)(cid:34)(cid:42)(cid:40)(cid:10)(cid:2615) (cid:2488)(cid:1157)(cid:1843) (cid:2230)(cid:1576)(cid:3294)(cid:1495) (cid:1238)(cid:1789)(cid:2021) (cid:2049)(cid:3319)(cid:3365)(cid:2190)"""
    if string not in text:
        return text
    else: return ""

def text_preprocessing_func(text):
    text = re.sub(r'\n[\n ]+', '\n', text)
    text = remove_table_of_contents(text)
    text = remove_duplicate_chars(text)
    text = remove_special_characters(text)
    text = remove_number_point(text)
    text = remove_new_line_character(text)
    text = remove_number_special_characters(text)
    text = remove_number_page(text)
    return text

if __name__ == '__main__':
    df = pd.read_json('./data/dataset-term-summary.json', encoding='utf-8')
    text = df['text'][0]
    result = text_preprocessing_func(text)

    print(text, result, sep='\n' + '-'*150 + '\n')
    
