#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the necessary library
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# In[23]:


#importing the dataset
df = pd.read_csv('testData.csv')


# In[24]:


#showing the data set
df


# In[25]:


#shape of the data set
df.shape


# In[26]:


#checking the null value by boolean 
df.isnull().any()


# In[27]:


#cheacking the null value by summation
df.isnull().sum()


# In[28]:


#defining independent value
x=df.iloc[:,:-1]
x


# In[29]:


#defining dependent value
y=df.iloc[:,1]
y


# In[30]:


#training and testing
from sklearn.model_selection import train_test_split


# In[31]:


xtrain,xtest,ytrain,ytest = train_test_split (x,y,test_size=0.30,random_state=1)


# In[32]:


xtrain


# In[33]:


#importing LinearRegression
from sklearn.linear_model import LinearRegression


# In[40]:


#declaring a object
Reg=LinearRegression()
Reg.fit(xtrain,ytrain)
Reg.score(xtest,ytest)


# In[42]:


#prediction for single value
Reg.predict([[50]])


# In[43]:


#coefficient
Reg.coef_


# In[47]:


#best fit line
plt.scatter(df['Area'],df['Price'],marker='+',color='red')
plt.xlabel('Area')
plt.ylabel('price')
plt.title('Home Prices in Dhaka city')
plt.plot(df.Area,Reg.predict(df[['Area']]))


# In[ ]:




