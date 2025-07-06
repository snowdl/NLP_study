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


# In[7]:


df = pd.read_csv('../../12_data/USA_Housing.csv')


# In[28]:


df.head()


# In[29]:


df.info()


# In[30]:


df.describe()


# In[31]:


df.columns


# In[32]:


sns.pairplot(df)


# In[33]:


sns.displot(df['Price'])


# In[34]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# In[35]:


df.columns


# In[36]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']]


# In[37]:


X = X.drop(columns=['Address'])


# In[ ]:





# In[39]:


y = df['Price']


# In[40]:


#Importing train_test_split from scikit-learn
from sklearn.model_selection import train_test_split


# In[42]:


# train_test_split

# What it does:train_test_split` is a function used to split your dataset into two parts:
#Training set**: Used to train the machine learning model.
#Testing set**: Used to evaluate the model's performance on unseen data.


train_test_split


# In[47]:


#test_size=0.4: 40% of the data will be used for testing, and 60% for training.
#random_state=101: Sets a seed for reproducibility. You’ll get the same split every time you run the code.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[48]:


#imports the LinearRegression class from the linear_model module of scikit-learn.
from sklearn.linear_model import LinearRegression


# In[49]:


lm = LinearRegression()


# In[50]:


lm.fit(X_train, y_train)


# In[51]:


#Printing the Intercept of the Linear Regression Model
print(lm.intercept_)


# In[52]:


lm.coef_


# In[51]:


X_train.columns


# In[54]:


cdf= pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# In[55]:


cdf.head()


# In[53]:


#Importing California Housing Dataset Loader
from sklearn.datasets import fetch_california_housing


# In[59]:


housing = fetch_california_housing()


# In[60]:


print(housing.data.shape)


# In[54]:


#Making Predictions with the Linear Regression Model
predictions = lm.predict(X_test)


# In[55]:


predictions


# In[56]:


y_test


# In[58]:


plt.scatter(y_test,predictions)


# In[65]:


sns.displot(y_test-predictions)


# In[59]:


#mporting Metrics Module from scikit-learn
from sklearn import metrics


# In[60]:


#Regression Evaluation Metrics

#three common evaluation metrics for regression problems:

#Mean Absolute Error (MAE) is the mean of the absolute value of the errors:
#Mean Squared Error (MSE) is the mean of the squared errors:
#Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

#Comparing these metrics:


#MAE is the easiest to understand, because it's the average error.
#MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
#All of these are loss functions, because we want to minimize them.


# In[61]:


#Calculating Mean Absolute Error (MAE)

metrics.mean_absolute_error(y_test,predictions)


# In[62]:


metrics.mean_squared_error(y_test, predictions)


# In[63]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[64]:


from sklearn import metrics


# In[65]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




