#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('../../data/Classified Data.csv', index_col=0)


# In[5]:


df.head()


# In[6]:


#imports the StandardScaler class from Scikit-learn’s preprocessing module.
#StandardScaler : Standardize the feature in the dataset 
from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


#df.drop('TARGET CLASS', axis=1) -> Removes the column 'TARGET CLASS', which is the label, so only feature columns are used.
#why not include 'Target class' -> the target variable (label) — not a feature.so it does not make sense and would leak label information

scaler.fit(df.drop('TARGET CLASS', axis=1))


# In[9]:


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[10]:


scaled_features


# In[11]:


df.columns


# In[12]:


df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])


# In[13]:


df_feat.head()


# In[14]:


# imports the train_test_split function from Scikit-learn’s model_selection module.
from sklearn.model_selection import train_test_split


# In[15]:


X =df_feat
y = df['TARGET CLASS']

#X_train, y__train --> 70% of the data → used to train the model
#X_test, y_test -->30% of the data → used to evaluate the model
#random_state=101 --->ensures that the split is reproducible — you'll get the same result every time you run it.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:


#imports the KNeighborsClassifier class from Scikit-learn’s neighbors module.

from sklearn.neighbors import KNeighborsClassifier


# In[17]:


#creates a K-Nearest Neighbors classifier and sets the number of neighbors (k) to 1.
#n_neighbors=1 ---> means the model will look at only the closest neighbor when making a prediction.


knn = KNeighborsClassifier(n_neighbors=1)


# In[70]:


knn.fit(X_train, y_train)


# In[71]:


params = knn.get_params()


# In[72]:


print(params)


# In[73]:


pred = knn.predict(X_test)


# In[74]:


pred


# In[75]:


from sklearn.metrics import classification_report, confusion_matrix


# In[76]:


print(confusion_matrix(y_test,pred))
print (classification_report(y_test, pred))


# In[18]:


#error_rate = [] -->Create an empty list to store the error rate for each value of k.
#for i in range(1, 40):	Loop through different k values from 1 to 39.
#knn = KNeighborsClassifier(n_neighbors=i) Create a KNN model with the current k value.
#knn.fit(X_train, y_train) -->Train the model on the training data.
#pred_i = knn.predict(X_test)	-->Predict the test labels using the current model.
#np.mean(pred_i != y_test) -->Compute the proportion of incorrect predictions (error rate).
#error_rate.append(...) -->	Add the current error rate to the list.

error_rate = []   
for i in range (1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[19]:


plt.figure(figsize=(10,6))  #Sets the size of the plot to 10 inches wide and 6 inches tall.

#range(1,40)  : the x-axis (values of k)
#error_rate   : the y-axis (corresponding error rates)

plt. plot(range(1,40), error_rate, color='blue', linestyle = 'dashed', marker ='o', markerfacecolor = 'red', markersize=10)
plt.title('Error rate V K value')
plt.xlabel('K')
plt.ylabel('Error rate')


# In[86]:


knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print (classification_report(y_test, pred))


# In[ ]:




