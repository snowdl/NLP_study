#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#mports the load_breast_cancer function from scikit-learn's datasets module
from sklearn.datasets  import load_breast_cancer


# In[17]:


#binary classification 
#indicating whether the tumor is malignant or benign

cancer = load_breast_cancer()


# In[18]:


print(cancer['DESCR'])


# In[19]:


df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])


# In[20]:


df.head()


# In[21]:


cancer['target_names']


# In[29]:


#imports the StandardScaler class from scikit-learn
#standardize features by removing the mean and scaling to unit variance

from sklearn.preprocessing import StandardScaler


# In[30]:


scaler = StandardScaler()


# In[31]:


scaler.fit(df)


# In[36]:


#transforming the data so that each feature has zero mean and unit variance.
scaled_data = scaler.transform(df)


# In[37]:


#PCA


# In[38]:


#imports the PCA (Principal Component Analysis) class from scikit-learn’s decomposition module
#used for reducing the dimensionality of data by projecting it onto principal components.

from sklearn.decomposition import PCA


# In[39]:


#creates a PCA object configured to reduce the dataset to 2
pca= PCA(n_components = 2)


# In[40]:


pca.fit(scaled_data)


# In[41]:


#x_pca = the PCA transformation to the scaled data, projecting it into a lower-dimensional space defined by the principal components
x_pca = pca.transform(scaled_data)


# In[22]:


scaled_data.shape


# In[28]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('First Prinicipal Component')
plt.ylabel('Second Principal Component')


# In[ ]:




