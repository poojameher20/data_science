#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
from sklearn.datasets import load_boston
boston_dataset = load_boston()

#%%
print(boston_dataset.keys())
#colnames=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(df)

#%%

print(df.head())

#%%
print(np.shape(df))

#%%
print(df.shape)
#%%
print(df.describe())

#%%
print(df.isnull().sum())

#%%
'''
Exploratory Data Analysis is a very important step before training the model.
In this section, we will use some visualizations to understand the relationship of the target variable with other features.
 '''
#sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['MEDV'], bins=30)
plt.show()

#%%
corr_matrix=df.corr().round(2)
#annot=True to print values inside the square
sns.heatmap(data=corr_matrix,annot=True)


#%%
#from above correlatio matrix we visualize features with target variable
features=['LSTAT','RM']
target=df['MEDV']

for i ,col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=df[col]
    y=target
    plt.scatter(x,y,marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
#%%
#preparing data for model

X=pd.DataFrame(np.c_[df['LSTAT'],df['RM']],columns=['LSTAT','RM'])    
Y=df['MEDV']

#%%
#spliting the data into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

#%%
#testing and training the model
from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)

#%%
#evalulate the model

from sklearn.metrics import mean_squared_error,r2_score
y_train_predict=lin_model.predict(x_train)
rmse=(np.sqrt(mean_squared_error(y_train,y_train_predict)))
r2=r2_score(y_train, y_train_predict)
print("model evalualtion for training dataset")
print("rmse={}".format(rmse))
print("r2_score={}".format(r2))

y_test_predict=lin_model.predict(x_test)
rmse=(np.sqrt(mean_squared_error(y_test,y_test_predict)))
r2=r2_score(y_test,y_test_predict)

print("\n")
print("model evaluation for testing datset")
print("rmse={}".format(rmse))
print("r2_score={}".format(r2))