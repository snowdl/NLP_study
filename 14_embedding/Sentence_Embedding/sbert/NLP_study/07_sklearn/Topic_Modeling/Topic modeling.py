#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load dataset (📁 경로 수정)
quora = pd.read_csv('../../12_data/quora_questions.csv')
print(quora.head())

# Check a sample question
print(quora['Question'][0])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(quora['Question'])  # Document-Term Matrix

# NMF Topic Modeling
nmf_model = NMF(n_components=20, random_state=42, init='nndsvd')  # init 추가 권장
nmf_model.fit(dtm)

# Print top 15 words for each topic
for index, topic in enumerate(nmf_model.components_):
    print(f"\n📌 Topic #{index}:")
    top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
    print(top_words)

# Assign most dominant topic to each document
topic_results = nmf_model.transform(dtm)
quora['Topic'] = topic_results.argmax(axis=1)

# Display updated dataframe
print(quora.head())

# Generate WordClouds for each topic
for index, topic in enumerate(nmf_model.components_):
    top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-20:]]
    weights = topic[topic.argsort()[-20:]]
    word_freq = dict(zip(top_words, weights))

    wc = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.title(f"Topic {index} WordCloud")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Topic distribution
print("\n🔍 Topic distribution for document 0:")
print(np.round(topic_results[0], 3))

topic_counts = quora['Topic'].value_counts()
print("\n📊 Document count per topic:")
print(topic_counts)

# Top document for each topic
print("\n📌 Top question per topic:")
for topic_num in range(nmf_model.n_components):
    top_doc_index = topic_results[:, topic_num].argmax()
    print(f"\n▶ Topic {topic_num}:")
    print(quora['Question'].iloc[top_doc_index])
