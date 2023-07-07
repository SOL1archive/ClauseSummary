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

    return text

def remove_new_line_character(text:str):
    text = re.sub(r"\n+", "", str(text))
    text = re.sub(r"\.", ".\n", str(text))

    return text

if __name__ == "__main__":
    text = "senteeeencccccceeee··············.\n\ndkfahksdglknasdkg.\n\n\n.dalkg\ndlfahadsf.\n안녕.\nㅏ안녕안녕\n\n안녕.\n\n" 
    ret = remove_duplicated_chars(text)
    ret2 = remove_special_characters(text)
    ret3 = remove_new_line_character(text)
    print(ret3)
    #print(ret2)