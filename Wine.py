# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:28:13 2020

@author: Ravi
"""


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd

data=load_wine()
df_wine=pd.DataFrame(data['data'],columns=data['feature_names'])
df_wine.head()
df_wine['target']=data['target']

x=data['data']
y=data['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

model=RandomForestClassifier(n_estimators=150)
model.fit(x_train,y_train)
pred=model.predict(x_test)
ac=accuracy_score(y_test,pred)
print(ac)

cm=confusion_matrix(y_test,pred)
print(cm)
sns.heatmap(cm,annot=True,cmap='Blues_r',yticklabels=['class_0','class_1','class_2'],xticklabels=['class_0p','class_1p','class_2p'])
plt.xlabel('Predicted values')
plt.ylabel('Original Values')
title='Accuracy- {}'.format(ac)
plt.title(title)

imp_feat=pd.Series(model.feature_importances_,index=data['feature_names'])
print(imp_feat)

plt.bar(data['feature_names'],imp_feat) #we can see every feature is contributing something in this algo.

accuracy=[]
for n_estimator in range(100,401,50):
    model=RandomForestClassifier(n_estimators=n_estimator)
    model.fit(x_train,y_train)
    p=model.predict(x_test)
    ac=accuracy_score(y_test,pred)
    accuracy.append(ac)

plt.bar(range(100,401,50),accuracy)
print(accuracy)
