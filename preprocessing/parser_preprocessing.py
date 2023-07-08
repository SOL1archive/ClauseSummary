import re
import pandas as pd


def remove_duplicated_chars(text:str): 
    result = "" 
    result += text[0]
    for i in range(1, len(text)): 
        if text[i - 1] != text[i]: 
            result += text[i] 
    
    return result 

def remove_special_characters(text:str):
    text = re.sub(r"·+",r" ", str(text))
    text = re.sub(r"●", "", str(text))
    text = re.sub(r"△", "", str(text))
    text = re.sub(r"】", "", str(text))
    text = re.sub(r"【", "", str(text))
    text = re.sub(r"-/d+-", "", str(text))

    return text

def remove_new_line_character(text:str):
    text = re.sub(r"\n+", "", str(text))
    text = re.sub(r"\.", ".\n", str(text))

    return text

def remove_number_special_characters(text:str):
    text_list = ['\u2460', '\u2461', '\u2462', '\u2463', '\u2464', '\u2465', '\u2466', '\u2467', '\u2468', '\u2469', '\u246A', '\u246B', '\u246C', '\u246D', '\u246E', '\u246F', '\u2470', '\u2471', '\u2472', '\u2473', '\u3251', '\u3252', '\u3253', '\u3254', '\u3255', '\u3256', '\u3257', '\u3258', '\u3259', '\u325A', '\u325B', '\u325C', '\u325D', '\u325E', '\u325F', '\u32B1', '\u32B2', '\u32B3', '\u32B4', '\u32B5', '\u32B6', '\u32B7', '\u32B8', '\u32B9', '\u32BA', '\u32BB', '\u32BC', '\u32BD', '\u32BE', '\u32BE']
    text_pattern = '|'.join(re.escape(char) for char in text_list)
    text = re.sub(text_pattern, "", str(text))

    return text

def remove_number_page(text:str):
    text = re.sub(r"\d+p", "", str(text))

    return text

#숫자뒤에 .이 오는 것을 제거
def remove_number_point(text:str):
    text = re.sub(r"(\d+)\.", r"\1: ", str(text))

    return text

def text_preprocessing_func(text):
    text = remove_duplicated_chars(text)
    text = remove_special_characters(text)
    text = remove_number_special_characters(text)
    text = remove_number_point(text)
    text = remove_new_line_character(text)
    text = remove_number_page(text)

    return text

if __name__ == "__main__":
    text = "senteeeenccccc】】ceeee··【①⑮【············.\ndkfahksdg△△△△lknasdkg.\n\n\n.dalkg\ndlfahadsf.\n안녕.\nㅏ안녕안녕\n\n안녕.\n\n●●●●●●●● 1.\ndsaflklkg\n2.\ndafasfd 46p 234123p 4521p 36p"
    result = text_preprocessing_func(text)
    print(text, result, sep='\n' + '-'*150 + '\n')
    #print(ret2)