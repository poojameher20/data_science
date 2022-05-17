
#%%
import pandas as pd
import numpy as np

#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\StudentsPerformances.csv")
print(df)
#%%
print(df.head(5))

#%%
print(df.describe())

#%%
print(df.isnull().sum())

#%%
print("dim",df.ndim)
print(df.shape)
print(df.size)

#%%
#data normalization 
df["reading score"]=df["reading score"]/df["reading score"].max()
print(df.head(5))
#%%
print(df.dtypes)
#%%
#type conversion
df[['math score']]=df[['math score']].astype("float")
df[['test preparation course']]=df[["test preparation course"]].astype("bool")
print(df.dtypes)

#%%
#convert categorical variable into qutative variable
mytest=pd.get_dummies(df['gender'])
print(mytest)


