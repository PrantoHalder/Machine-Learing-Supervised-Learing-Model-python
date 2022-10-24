#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as splt
import seaborn


# In[3]:


df=pd.read_csv('heart.csv')
df


# In[4]:


df.shape


# In[5]:


df['target'].value_counts()


# In[6]:


seaborn.countplot(df['target'])


# In[7]:


df.isnull().values.any() 


# In[8]:


df.isna().sum()


# In[9]:


splt.figure(figsize=(25,10))
fig=seaborn.countplot(x='age',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Age of the Patients")
splt.ylabel("Counts")
splt.title("Age vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[10]:



seaborn.countplot(x='sex',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Gender of the Patients ")
splt.ylabel("Counts")
splt.title("Gender vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[11]:



seaborn.countplot(x='cp',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Chest Pain")
splt.ylabel("Values")
splt.title("Chest Pain vs Cardivascular diseases ") # You can comment this line out if you don't need title
splt.show(fig)


# In[12]:


splt.figure(figsize=(30,10))
seaborn.countplot(x='trestbps',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Resting Blood Pressure")
splt.ylabel("Values")
splt.title("Resting Blood Pressure vs Cardivascular diseases ") # You can comment this line out if you don't need title
splt.show(fig)


# In[13]:



seaborn.countplot(x='fbs',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Fasting blood pressure")
splt.ylabel("Values")
splt.title("Fasting blood pressure vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[14]:



seaborn.countplot(x='restecg',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Resting electrographic measurement.")
splt.ylabel("Values")
splt.title("Resting electrographic measurement vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[15]:





seaborn.countplot(x='slope',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Peak exercise ST segment")
splt.ylabel("Values")
splt.title("Peak exercise ST segment vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[16]:



seaborn.countplot(x='ca',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Coronary Calcium scan")
splt.ylabel("Values")
splt.title("Coronary Calcium scan vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[17]:



seaborn.countplot(x='thal',hue='target',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1),)
splt.xlabel("Thalassemia.")
splt.ylabel("Values")
splt.title("Thalassemia vs Cardivascular diseases") # You can comment this line out if you don't need title
splt.show(fig)


# In[18]:


splt.figure(figsize=(16,10))

splt.xticks(fontsize=14)
splt.yticks(fontsize=14)
seaborn.heatmap(df.corr(),annot=True)
splt.show()


# In[28]:


x=df.iloc[:,:-1]
x


# In[20]:


y=df.iloc[:,13]
y


# In[21]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)
xtrain


# In[40]:


from sklearn.linear_model import LinearRegression
clf2=LinearRegression()
clf2.fit(xtrain,ytrain)
clf2.score(xtest,ytest)


# In[54]:


import numpy as np
import matplotlib.pyplot as plt

y = np.array([97.77,98.87,93.54,96.77])
x = np.array(['Random Forest','AdaBoost','Decision Tree','SVM'])

plt.title("Algoritms predicted value")
plt.xlabel("Algorithms Used")
plt.ylabel("Perdicted percentile using the algorihtm")

plt.plot(x, y)

plt.grid()

plt.show()


# In[55]:





# In[ ]:




