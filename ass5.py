'''Implement logistic regression using Python/R to perform classification on 
Social_Network_Ads.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall 
on the given dataset'''

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\Social_Network_Ads.csv")
print(df)

#%%
print(df.head())
#%%
print(df.describe())

#%%
print(df.isnull().sum())
#%%
#correlation matrix
sns.heatmap(df.corr().abs(),  annot=True)

#%%
#prepare data for model
X=pd.DataFrame(np.c_[df['Age'],df['EstimatedSalary']],columns=['Age','EstimatedSalary'])
Y=df['Purchased']
#%%
#spliting datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)
#%%
#testing and trainig model
from sklearn.linear_model import LogisticRegression
l_model=LogisticRegression()
l_model.fit(X_train,Y_train)
#%%
#evalutae the model
y_predict=l_model.predict(X_test)

#%%

from sklearn.metrics import confusion_matrix
c_mat=confusion_matrix(Y_test,y_predict)
print(c_mat)

#%%
sns.heatmap(c_mat,annot=True)
plt.show()

#%%
TP,FP,TN,FN=c_mat.ravel()
print("TP",TP)
print("FP",FP)
print("FN",FN)
print("TN",TN)

#%%
from sklearn .metrics import accuracy_score,precision_score,recall_score
print(accuracy_score(Y_test,y_predict))

#%%
print(precision_score(Y_test,y_predict))
print(recall_score(Y_test,y_predict))

