# Machine-Learing-Supervised-Learing-Mdel
This Code shows how we can split ,test, train and predict
#Adding Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as splt
import seaborn

#importing dataset
df=pd.read_csv('/dengue.csv')#,sep=';')
df

#condition of the data
df.shape
df['dengue.dengue'].value_counts()

#Target value condition
seaborn.countplot(df['dengue.dengue'])  #target value condition

#Null Value Calculation
df.isnull().values.any()  #null value check
df.isna().sum()

#Corr before handling null value 
df=df.drop(['id'],axis=1)
df
df.corr()
splt.figure(figsize=(16,10))

#correlation Heatmaap
splt.xticks(fontsize=14)
splt.yticks(fontsize=14)
seaborn.heatmap(df.corr(),annot=True)
splt.show()

#round a value
seaborn.countplot(x='gender',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
df['yr']=(df['age']/365).round(0)
df['yr']
splt.figure(figsize=(16,10))
seaborn.countplot(x='yr',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
df.describe()

#droping a value
df=df.drop(['yr'],axis=1)
df=df.drop(['id'],axis=1)
df
df.corr()
splt.figure(figsize=(16,10))

splt.xticks(fontsize=14)
splt.yticks(fontsize=14)
seaborn.heatmap(df.corr(),annot=True)
splt.show()
df=df.drop(['yr'],axis=1)
df=df.drop(['id'],axis=1)
df
x=df.iloc[:,:-1]
x
y=df.iloc[:,11]
y

#split test train
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)
xtrain

#rabdom forest
from sklearn.ensemble import RandomForestClassifier
Rclf=RandomForestClassifier()
Rclf.fit(xtrain,ytrain)
Rclf.score(xtest,ytest)

#Decision
from sklearn import tree
Clf = tree.DecisionTreeClassifier()  
Clf.fit(xtrain,ytrain)
Clf.score(xtest,ytest)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)
from sklearn.svm import SVC
model=SVC(gamma='auto')
model.fit(xtrain,ytrain)
model.score(xtest,ytest)
