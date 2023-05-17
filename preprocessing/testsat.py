import re

def remove_number_dot_from_string(str):
    pattern = r'\b을\b'
    return re.sub(pattern, r'', str)

# 예시
string_with_number_dot = "을오징어을 오징어 을 오징어"
result = remove_number_dot_from_string(string_with_number_dot)
print(result)