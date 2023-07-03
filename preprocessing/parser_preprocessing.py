def remove_duplicated_chars(characters): 
    result = "" 
    result += characters[0]
    for i in range(1, len(characters)): 
        if characters[i - 1] != characters[i]: 
            result += characters[i] 
    
    return result 

if __name__ == "__main__":
    characters = "senteeeencccccceeee" 
    ret = remove_duplicated_chars(characters)