#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


#Euclidean distance


# In[2]:


a = np.array([1,0,-1,6,8])
b = np.array([0,11,4,7,6])


# In[3]:


d = np.linalg.norm(a-b)


# In[4]:


print("this is my first eudclidean distance : ", d)


# In[5]:


#cosine similarity


# In[6]:


cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos_sim)


# In[8]:


#using Scipy


# In[9]:


from scipy import spatial


# In[10]:


cos_sim = 1 - spatial.distance.cosine(a, b)
print(cos_sim)


# In[11]:


#using sklearn


# In[12]:


from sklearn.metrics.pairwise import cosine_similarity


# In[14]:


a = np.array([1, 0, -1, 6, 8]).reshape(1, -1)
b = np.array([0, 11, 4, 7, 6]).reshape(1, -1)

cos_sim = cosine_similarity(a, b)
print(cos_sim[0][0])


# In[ ]:




