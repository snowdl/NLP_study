#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


quora = pd.read_csv('quora_questions.csv')


# In[7]:


quora.head()


# In[8]:


quora.Question[0]


# In[9]:


#pre processing


# In[10]:


#Import the TfidfVectorizer class from scikit-learn= converts text doc into a matrix TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


'''
max_df=0.95 =  Ignore terms that appear in more than 95% of documents (too common, likely uninformative).
min_df=2 = Ignore terms that appear in fewer than 2 documents (too rare)
stop_words='english' = Remove common English stop words like "the", "is", "and", etc.

'''
tfidf = TfidfVectorizer(max_df = 0.95, min_df=2, stop_words='english')


# In[12]:


'''
Learn the vocabulary and IDF from the "Question" column of the quora DataFrame.
Transform the questions into a Document-Term Matrix (DTM) where each row is a document, each column is a term, and the values are TF-IDF scores
'''

dtm = tfidf.fit_transform(quora['Question']) 


# In[13]:


#outputs the sparse matrix representing the TF-IDF weighted document-term matrix.
dtm  


# In[15]:


#Non-negative Matrix Factorization


# In[16]:


from sklearn.decomposition import NMF


# In[17]:


# Create an NMF model to extract 20 topics from the document-term matrix
nmf_model = NMF(n_components=20, random_state=42)


# In[19]:


# Fit the NMF model on the document-term matrix (dtm)
nmf_model.fit(dtm)


# In[20]:


for index, topic in enumerate(nmf_model.components_):
    print(f"The top 15 words for topic #{index}")

#Get the indices of the top 15 words with highest weights in this topic,
# then retrieve their corresponding feature names (words) from the TF-IDF vectorizer
    print ([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[21]:


#applies the trained NMF model to the document-term matrix (dtm)
topic_results = nmf_model.transform(dtm)


# In[22]:


#argmax(axis=1) returns the topic index that has the largest value for each document.
quora['Topic'] = topic_results.argmax(axis=1)


# In[25]:


quora.head()


# In[27]:


#Visualization and analysis of the top words in each topic


# In[23]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[25]:


# Loop over each topic extracted by NMF
for index, topic in enumerate(nmf_model.components_):
    # Get top 20 words with highest importance for this topic
    top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-20:]]

    # Get corresponding weights (importance scores) for these top words
    weights = topic[topic.argsort()[-20:]]

    # Create a dictionary mapping words to their weights
    word_freq = dict(zip(top_words, weights))

    # Generate a word cloud image from the word frequencies
    wc = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)

    # Set up matplotlib figure size
    plt.figure(figsize=(10,5))

    # Set the title of the plot as the topic number
    plt.title(f"Topic {index} WordCloud")

    # Display the generated word cloud image
    plt.imshow(wc, interpolation="bilinear")

    # Remove axis for better visualization
    plt.axis("off")

    # Show the plot
    plt.show()


# In[28]:


#Analyzing topic distributions across documents and aggregating document counts per topic


# In[34]:


# Print the topic distribution scores for the first document (index 0)
print(np.round(topic_results[0], 3))

# Count how many documents are assigned to each topic (most dominant topic)
topic_counts = quora['Topic'].value_counts()

# Print the counts of documents per topic
print(topic_counts)


# In[35]:


#Identifying top documents for each topic"


# In[36]:


# Iterate through each topic index
for topic_num in range(nmf_model.n_components):
    # Get all document scores for the current topic
    topic_column = topic_results[:, topic_num]

    # Find the index of the document with the highest score for this topic
    top_doc_index = topic_column.argmax()

    # Print topic number and its representative document (question)
    print(f"Top document for topic {topic_num}:")
    print(quora['Question'].iloc[top_doc_index])
    print()


# In[ ]:




