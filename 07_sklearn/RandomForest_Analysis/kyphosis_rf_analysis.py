#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Load the kyphosis dataset from the 12_data folder
df = pd.read_csv('../../12_data/📓_[0623]_kyphosis.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


sns.pairplot(df, hue='Kyphosis')


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X = df.drop('Kyphosis', axis=1)


# In[10]:


y = df['Kyphosis']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)


# In[12]:


#DecisionTreeClassifier--> a tree-based machine learning algorithm that splits data based on feature values and is commonly used for classification tasks.

from sklearn.tree import DecisionTreeClassifier


# In[13]:


#creates a Decision Tree classifier model using the default parameters.
dtree = DecisionTreeClassifier()


# In[14]:


dtree.fit(X_train, y_train)


# In[15]:


params = dtree.get_params()


# In[16]:


params


# In[17]:


predictions = dtree.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix


# In[19]:


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# In[19]:


# imports the Random Forest Classifier from Scikit-learn
#Random Forest : an ensemble of many decision trees.
# is trained on a random subset of data and features.
#The final prediction is made by majority voting (classification) or averaging (regression).
#reduces overfitting and improves generalization performance.

from sklearn.ensemble import RandomForestClassifier; from sklearn.metrics import accuracy_score



# In[20]:


rfc = RandomForestClassifier(n_estimators=200)
#Creates a random forest classifier with 200 trees.
#Increasing the number of trees can improve performance and stability, but may also increase training time.

rfc.fit(X_train, y_train)


# In[22]:


# Label Encoding (Kyphosis: 'absent'/'present' → 0/1)
#The target variable y contains categorical labels 'absent' and 'present'.
# use LabelEncoder to convert these string labels into numeric values (0 for 'absent' and 1 for 'present')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[23]:


#Data splitting
#70% of the data is used for training (X_train, y_train) and 30% for testing (X_test, y_test).
#The random_state=42 ensures reproducibility of the split.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


#model Training
# create a Random Forest classifier with 200 trees (n_estimators=200).
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train, y_train)


# In[ ]:





# In[25]:


#predictions
y_pred = rfc.predict(X_test)


# In[ ]:





# In[27]:


# Accuracy Evaluation:alculate the prediction accuracy by comparing the predicted labels (y_pred) with the true labels (y_test) using accuracy_score.
acc = accuracy_score(y_test, y_pred)
print("Accuracy Evaluation:", acc)


# In[ ]:


#Feature Importance Visualization


# In[28]:


#Extract the importance of each feature using the model’s feature_importances_ attribute.
importances = rfc.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance (Kyphosis)")
plt.show()


# In[22]:


params = rfc.get_params()
print(params)

rfc_pred = rfc.predict(X_test)  # 예측 실행
# In[25]:


print(X_test)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[27]:


rfc_pred = rfc.predict(X_test)


# In[28]:


print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))


# In[28]:


rfc = RandomForestClassifier()  # 모델 생성
rfc.fit(X_train, y_train) 


# In[ ]:


im


# In[ ]:





# In[ ]:





# In[ ]:




