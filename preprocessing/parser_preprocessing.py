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
    text_list = ['\nU+2460', '\nU+2461', '\nU+2462', '\nU+2463', '\nU+2464', '\nU+2465', '\nU+2466', '\nU+2467', '\nU+2468', '\nU+2469', '\nU+246A', '\nU+246B', '\nU+246C', '\nU+246D', '\nU+246E', '\nU+246F', '\nU+2470', '\nU+2471', '\nU+2472', '\nU+2473', '\nU+3251', '\nU+3252', '\nU+3253', '\nU+3254', '\nU+3255', '\nU+3256', '\nU+3257', '\nU+3258', '\nU+3259', '\nU+325A', '\nU+325B', '\nU+325C', '\nU+325D', '\nU+325E', '\nU+325F', '\nU+32B1', '\nU+32B2', '\nU+32B3', '\nU+32B4', '\nU+32B5', '\nU+32B6', '\nU+32B7', '\nU+32B8', '\nU+32B9', '\nU+32BA', '\nU+32BB', '\nU+32BC', '\nU+32BD', '\nU+32BE', '\nU+32BE',]
    text_pattern = '|'.join(re.escape(char) for char in text_list)
    text = re.sub(text_pattern, "", str(text))

    return text

def remove_number_page(text:str):
    text = re.sub(r"\d+p", "", str(text))

    return text

if __name__ == "__main__":
    text = "senteeeenccccc】】ceeee··【【············.\n\ndkfahksdg△△△△lknasdkg.\n\n\n.dalkg\ndlfahadsf.\n안녕.\nㅏ안녕안녕\n\n안녕.\n\n●●●●●●●●" 
    ret = remove_duplicated_chars(text)
    ret2 = remove_special_characters(text)
    ret3 = remove_new_line_character(text)
    print(ret2)
    #print(ret2)