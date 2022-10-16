#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import numpy


# In[3]:


df = pd.read_csv('car driving risk analysis.csv')


# In[4]:


df


# In[5]:


df.shape


# In[8]:


df.isnull().any()


# In[9]:


df.isnull().sum()


# In[12]:


plt.scatter(df['speed'],df['risk'])
plt.xlabel('Speed of the car')
plt.ylabel('Risk factor')
plt.title('Rsik analysis of a car based on Speed')


# In[14]:


x=df.iloc[:,:-1]
x


# In[15]:


y=df.iloc[:,1]
y


# In[16]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split (x,y,test_size=0.30,random_state=1)
from sklearn.linear_model import LinearRegression
Reg=LinearRegression()
Reg.fit(xtrain,ytrain)
Reg.score(xtest,ytest)


# In[18]:


plt.scatter(df['speed'],df['risk'])
plt.xlabel('Speed of the car')
plt.ylabel('Risk factor')
plt.title('Rsik analysis of a car based on Speed')
plt.plot(df.speed,Reg.predict(df[['speed']]))


# In[ ]:




