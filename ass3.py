#%%
import pandas as pd

#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\mall.csv")
print(df)

#%%
print(df.head(5))

#%%
#summary
print(df.info())
#%%
print(df.describe())

#%%
print("shape",df.shape)

#%%
df.groupby(('Age'))['Annual Income (k$)'].mean()
df.groupby(('Age'))['Annual Income (k$)'].describe()
#%%
range=[0,20,30,40,50,60]
print(df.groupby(pd.cut(df.Age,range))['Annual Income (k$)'].describe())

#%%
'''
Write a Python program to display some basic statistical 
details like percentile, mean, standard deviation etc. of the 
species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of iris.csv
 dataset.
'''

#%%
df2=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\iris.csv")
print(df2)

#%%
print(df2.head())

#%%
setosa=df2['species']=="Iris-setosa"
setosa=df2[setosa].describe()
print(setosa)

#%%
vcolor=df2['species']=="Iris-versicolor"
vcolor=df2[vcolor].describe()
print(vcolor)

#%%
