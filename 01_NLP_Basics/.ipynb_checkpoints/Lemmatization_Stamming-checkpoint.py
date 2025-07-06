#!/usr/bin/env python
# coding: utf-8

# In[13]:


from nltk.stem import WordNetLemmatizer

# Create an instance of the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# A list of words we want to lemmatize
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('Before lemmatization:',words)
print('After lemmatization:',[lemmatizer.lemmatize(word) for word in words])


# In[14]:


lemmatizer.lemmatize('dies', 'v')


# In[15]:


lemmatizer.lemmatize('watched', 'v')


# In[16]:


lemmatizer.lemmatize('has', 'v')


# In[22]:


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


# In[23]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()


# In[24]:


sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)


# In[25]:


print('Stemmming before :', tokenized_sentence)
print('Stemming after :',[stemmer.stem(word) for word in tokenized_sentence])


# In[27]:


words = ['formalize', 'allowance', 'electricical']

print('Stemmming before :',words)
print('Stemmming after:',[stemmer.stem(word) for word in words])


# In[29]:


#comparison btw PorterStemmer& LancasterStemmer


# In[ ]:


from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer  # 이 줄이 추가되어야 합니다

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('어간 추출 전 :', words)
print('PorterStemmer:', [porter_stemmer.stem(w) for w in words])
print('LancasterStemmer:', [lancaster_stemmer.stem(w) for w in words])


# In[ ]:




