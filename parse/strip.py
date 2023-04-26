import re

def strip(contents):
    clause = []
    for content in contents:
        content.pop(0) #delete contents table

        text = ""
        for page in content: #merge all page of a chapter
            text += page
            text = text[text.find("제1조"):]

    sentences = text.splitlines()

    articles = []
    for i in range(len(sentences)):
        if re.search("^제.{1,}조", sentences[i]):
            articles.append(sentences[i])

        else:
            articles[len(articles) - 1] += sentences[i]

    if len(articles) > 0:
        clause.append(articles)
    
    return clause
