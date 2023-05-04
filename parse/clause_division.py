import re

### function page2text ###
# 1. get param content_1D(1D array by [page]])
# 2. delete 1st page of content 
# 3. merge all page of content
# 4. return 1D array by [lines], text divided by enter(\n)
def page2text(content_1D):
    content_1D.pop(0) #delete contents table
    
    text = ""
    for page in content_1D: #for all pages
        text += page #merge pages

    return text.splitlines()

### function text2articles ###
#1. get param text_1D(1D array by [sentences])
#2. find elements of text_1D starts with "제1, 2, 3 ...조"
#3. if starts with "제n조", append to articles
#   - "제n조"로 시작하는 문장은 articles에 append
#   else, append to before previous article
#   - "제n조"로 시작하는 문장은 새로운 조항이 아닌 이전 조항
#   - 따라서 새로 append하지 않고 이전의 article에 내용을 추가

def text2articles(text_1D):
    articles_1D = []
    for i in range(len(text_1D)):
        if re.search("^제.{1,}조", text_1D[i]):
            articles_1D.append(text_1D[i])
        else:
            articles_1D[len(articles_1D)-1] += text_1D[i]

    return articles_1D
    
### function articles2clauses(main) ###
#1. get param contents(1D array by [page])
#2. for all contents,
#       call page2text func -> text_1D array
#       call text2articles func -> articles_1D array
#       if articles_1D array is not empty,
#           append to clauses_2D
# 3. clauses_2D array
#    - [목차][조항]으로 구성된 배열
#    - ex) [보통약관][제1조. ---], [보통약관][제2조. ---]

def articles2clauses(contents_1D):
    clauses_2D = []
    for content in contents_1D:
        text_1D = page2text(content)
        articles_1D = text2articles(text_1D)

        if articles_1D:
            clauses_2D.append(articles_1D)

    return clauses_2D
