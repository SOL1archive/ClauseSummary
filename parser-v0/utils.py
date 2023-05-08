import re
import PyPDF2

num_index_re = r'[1-9]+[0-9]*[.]'
kor_index_re = r'[가나다라마바사아자차카타파하][.]'
eng_index_re = r'[A-Z][.]'
roman_index_re = r'^(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)$'
index_re_lt = [
    num_index_re, kor_index_re, eng_index_re, roman_index_re
]
white_space_re = r'(\s*)'

url_re = r'(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/'

def find_header(text: str, symbol: str):
    idx_lt = []

    for idx, line in enumerate(text.splitlines()):
        if line.strip().startswith(symbol):
            idx_lt.append(idx)
        
    return idx_lt

def remove_header(text: str, symbol: str) -> str:
    text_lines = []
    header_idx_lt = find_header(text, symbol)
    for idx, text_line in enumerate(text.splitlines()):
        if idx not in header_idx_lt:
            text_lines.append(text_line)

    return '/n'.join(text_lines)

def remove_index(text: str) -> str:
    for index_re in index_re_lt:
        text = re.sub(index_re, '', text)

    return text

def map_lines(text: str, call) -> str:
    result_lt = []
    for line in text.splitlines():
        result_lt.append(call(line))

    return '\n'.join(result_lt)

def strip_lines(text: str) -> str:
    result_lt = []
    for line in text.splitlines():
        result_lt.append(line.strip())

    return '\n'.join(result_lt)

def remove_url(text: str) -> str:
    return re.sub(url_re, '', text)

def get_body(file, upper_limit, lower_limit) -> list:
    pages = []
    parts = []
    reader = PyPDF2.PdfReader(file)

    def body(text, cm, tm, font_dict, font_size):
        y = tm[5]
        if lower_limit < y < upper_limit:
            parts.append(text)

    for page in reader.pages:
        page.extract_text(visitor_text=body)
        pages.append(
            ''.join(parts)
        )
    
    return pages

def whitespace_match(source_text: str, target_text: str):
    target_text_re = white_space_re.join(list(target_text))
    match_lt = re.findall(target_text_re, source_text)

    return match_lt

def isin(source_text:str, target_text: str):
    return len(whitespace_match(source_text, target_text)) != 0

def is_pdf(filename: str):
    return filename.endswith('.pdf')
