import re
def text_preprocessing_func(text):
    return re.sub(r'\n[\n ]+', '\n', text)
