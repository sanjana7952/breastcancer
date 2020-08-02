# breastcancerprediction
#importing libraries
import pandas as pd
import numpy as np

#loading the dataset
df=pd.read_csv(r'C:\Users\Sanjana Singh\Downloads\breast_cancer.csv')

#reading the first five rows
df.head()

#frequency of diagnosis column
df['diagnosis'].value_counts()

#deleting the null and not useful values
df1=df.drop('Unnamed: 32',axis=1)
df2=df1.drop('id',axis=1)

#sum of null values
df2.isnull().sum()

#assinging M =1 and B=0
df2['diagnosis']=df2['diagnosis'].map({'M':1,'B':0})

#check if M and B has been replaced
df2.head()

#dropping the target varible
x=df2.drop('diagnosis',axis=1)
y=df2['diagnosis']

#importing train_test_split
from sklearn.model_selection import train_test_split

#implementing train_test_split to split the dataset
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

#importing logistic regression
from sklearn.linear_model import LogisticRegression

#creating object
logreg=LogisticRegression()

#fit the dataset
logreg.fit(train_x,train_y)

#check the accuracy
logreg.score(train_x,train_y)

logreg.fit(test_x,test_y)

logreg.score(test_x,test_y)
