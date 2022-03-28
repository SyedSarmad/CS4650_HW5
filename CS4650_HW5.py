# importing the nessasary libraries
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import en_core_web_lg
import pickle
import spacy
nlp_eng = spacy.load("en_core_web_lg")
news_api = NewsApiClient(api_key='b13ad550c31f41a8a73d636405c5918c')


def get_keywords_eng(token):
    keywords = []
    punctuation = string.punctuation

    for i in token:
        if i in nlp_eng.Defaults.stop_words or i in punctuation:
            continue
        else:
            keywords.append(i)
    return keywords


temp = news_api.get_everything(q='coronavirus', language='en', from_param='2022-02-30', to='2022-03-30', sort_by='relevancy')
filtered_articles = []
for i, article in enumerate(temp['articles']):
    title = article['title']
    description = article['description']
    content = article['content']
    date = article['publishedAt']
    filtered_articles.append({'title': title, 'date': date,'desc': description, 'content': content})
    df = pd.DataFrame(filtered_articles)
    df = df.dropna()
    df.head()

tokenizer = RegexpTokenizer(r'\w+')

results = []
for content in df.content.values:
    content = tokenizer.tokenize(content)
    results.append([x[0] for x in Counter(get_keywords_eng(content)).most_common(5)])

df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

df.to_csv('data.csv', index=False)
