#%%
'''
1. Extract Sample document and apply following document preprocessing methods:
Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2. Create representation of document by calculating Term Frequency and Inverse Document 
Frequency

'''

#%%
import pandas as pd
import numpy as np
#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\iris.csv")
print(df)
 
#%%
print(df.head())
#%%
print(df.shape)
#%%
print(df.describe())

#%%
print(df.isnull().sum())
#%%
print(df.columns)

#%%
import seaborn as sns
sns.boxplot(df['sepal_length'])


#%%
sns.boxplot(df['petal_length'])
#%%
sns.boxplot(df['sepal_width'])
#%%
print(df=df[~(df['sepal_width']<2.4)& ~(df['sepal_width']>4.0)])
#%%
sns.boxplot(df['petal_width'])
#%%

x=pd.DataFrame(np.c_[df['sepal_length'],df['sepal_width'],df['petal_length'],df['petal_width']])
y=df['species']

#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3)
#%%
from sklearn.naive_bayes import GaussianNB
gau=GaussianNB()

gau.fit(x_train,y_train)
y_pred=gau.predict(x_test)

#%%
from sklearn.metrics import confusion_matrix
cmat=confusion_matrix(y_test, y_pred)

cmat
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
#%%

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
r2=r2_score(y_test, y_pred)
print(r2)
