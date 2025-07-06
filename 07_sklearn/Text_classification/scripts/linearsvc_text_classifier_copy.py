ㅋ#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv('../../12_data/moviereviews.tsv', sep='\t')


# In[5]:


df.head()


# In[6]:


len(df)


# In[7]:


print(df['review'][2])


# In[8]:


df.isnull().sum()


# In[9]:


#df.dropna(inplace=True) removes all rows from the pandas DataFrame df that contain missing values (NA or NaN).
df.dropna(inplace=True)


# In[10]:


df.isnull().sum()


# In[11]:


mystring ='hello'
empty = ' '


# In[12]:


#isspace() method checks if a string consists only of whitespace characters (spaces, tabs, newlines).
empty.isspace()


# In[13]:


blanks = [i for i, lb, rv in df.itertuples(index=True) if rv.isspace()]


# In[14]:


blanks


# In[15]:


df.drop(blanks, inplace=True)


# In[16]:


len(df)


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X = df['review']


# In[19]:


y = df['label']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


#Pipeline: Allows chaining multiple steps (like preprocessing, feature extraction, and model training) into a single sequential pipeline.
#TfidfVectorizer: Converts raw text into TF-IDF vectors (Term Frequency-Inverse Document Frequency), which numerically represent the importance of words in the documents.
#LinearSVC: A linear Support Vector Classifier. It is a powerful linear model often used for text classification tasks.


# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# In[23]:


#step 1 :abeled 'tfidf', applies TfidfVectorizer() to convert raw text data into TF-IDF feature vectors.
#step 2 : abeled 'clf', applies a LinearSVC() classifier for training and prediction.
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])


# In[24]:


text_clf.fit(X_train, y_train)


# In[25]:


predictions = text_clf.predict(X_test)


# In[26]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[27]:


print(confusion_matrix, y_test, predictions)


# In[29]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Assume predictions are already made
# predictions = text_clf.predict(X_test)

# 1. Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# 2. Print the classification report (precision, recall, F1-score, etc.)
cr = classification_report(y_test, predictions)
print("\nClassification Report:")
print(cr)

# 3. Print the accuracy score
acc = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {acc:.4f}")


# In[ ]:




