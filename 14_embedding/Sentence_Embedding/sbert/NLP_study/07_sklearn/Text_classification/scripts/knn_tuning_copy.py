#!/usr/bin/env python
# coding: utf-8

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# imports the make_blobs function from the sklearn.datasets module.
#make_blobs : a utility function to generate synthetic datasets for clustering or classification tasks.
from sklearn.datasets import make_blobs


# In[12]:


#n_samples=200 -->Creates 200 data points in total.
#n_features = 2 -->data point has 2 features (making it 2-dimensional)
#centers =4 -->e points are grouped around 4 cluster centers
#cluster_std = 1.8 -->Each cluster has a standard deviation of 1.8
#random_state=101 --> Fixes the random seed
#will return
#data[0]: The generated feature data points (an array of shape (200, 2)).
#data[1]: The cluster labels for each data point (integers 0 to 3, indicating cluster membership)

data = make_blobs(n_samples = 200, n_features = 2, centers =4, cluster_std = 1.8, random_state =101)


# In[15]:


plt.scatter(data[0][:,0], data[0][:,1],c=data[1], cmap='rainbow')


# In[16]:


from sklearn.cluster import KMeans


# In[17]:


#creates an instance of the KMeans clustering algorithm with the number of clusters (k) set to 4.
#n_clusters=4 tells the algorithm to find 4 distinct clusters in the dataset
kmeans = KMeans(n_clusters=4)


# In[19]:


kmeans.fit(data[0])


# In[20]:


#kmeans.labels_ corresponds to the cluster index (from 0 to n_clusters-1) of the respective data point.
kmeans.labels_


# In[22]:


#returns an array containing the coordinates of the cluster centers found by the KMeans algorithm.
kmeans.cluster_centers_


# In[27]:


fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(10,6))

ax1.set_title ('K means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_, cmap = 'rainbow')
ax2.set_title ('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap = 'rainbow')


# In[ ]:




