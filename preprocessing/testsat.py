import re

def remove_number_dot_from_string(str):
    pattern = r'(\d+)\.'
    return re.sub(pattern, r'\n\1.', str)

# 예시
string_with_number_dot = "Hello123. World456.!"
result = remove_number_dot_from_string(string_with_number_dot)
print(result)
