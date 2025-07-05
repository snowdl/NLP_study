#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


import os
print(os.getcwd())


# In[31]:


df = pd.read_csv('../../../12_data/classified_./12_data/data.csv', index_col=0)


# In[32]:


df.head()


# In[33]:


#imports the StandardScaler class from Scikit-learn’s preprocessing module.
#StandardScaler : Standardize the feature in the dataset 
from sklearn.preprocessing import StandardScaler


# In[34]:


scaler = StandardScaler()


# In[35]:


#df.drop('TARGET CLASS', axis=1) -> Removes the column 'TARGET CLASS', which is the label, so only feature columns are used.
#why not include 'Target class' -> the target variable (label) — not a feature.so it does not make sense and would leak label information

scaler.fit(df.drop('TARGET CLASS', axis=1))


# In[36]:


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[37]:


scaled_features


# In[38]:


df.columns


# In[39]:


df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])


# In[40]:


df_feat.head()


# In[41]:


# imports the train_test_split function from Scikit-learn’s model_selection module.
from sklearn.model_selection import train_test_split


# In[42]:


X =df_feat
y = df['TARGET CLASS']

#X_train, y__train --> 70% of the data → used to train the model
#X_test, y_test -->30% of the data → used to evaluate the model
#random_state=101 --->ensures that the split is reproducible — you'll get the same result every time you run it.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[43]:


#imports the KNeighborsClassifier class from Scikit-learn’s neighbors module.

from sklearn.neighbors import KNeighborsClassifier


# In[44]:


#creates a K-Nearest Neighbors classifier and sets the number of neighbors (k) to 1.
#n_neighbors=1 ---> means the model will look at only the closest neighbor when making a prediction.


knn = KNeighborsClassifier(n_neighbors=1)


# In[45]:


knn.fit(X_train, y_train)


# In[46]:


params = knn.get_params()


# In[47]:


print(params)


# In[48]:


pred = knn.predict(X_test)


# In[49]:


pred


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix


# In[51]:


print(confusion_matrix(y_test,pred))
print (classification_report(y_test, pred))


# In[52]:


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


# In[53]:


plt.figure(figsize=(10,6))  #Sets the size of the plot to 10 inches wide and 6 inches tall.

#range(1,40)  : the x-axis (values of k)
#error_rate   : the y-axis (corresponding error rates)

plt. plot(range(1,40), error_rate, color='blue', linestyle = 'dashed', marker ='o', markerfacecolor = 'red', markersize=10)
plt.title('Error rate V K value')
plt.xlabel('K')
plt.ylabel('Error rate')


# In[54]:


knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print (classification_report(y_test, pred))


# In[ ]:





# In[ ]:





# In[ ]:




