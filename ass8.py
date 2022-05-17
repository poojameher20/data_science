#%%
import seaborn as sns

#%%
df=sns.load_dataset('titanic')

#%%
df.head(5)

#%%
df.shape

#%%
df.isnull().sum()

#%%
sns.distplot(df['fare'])

#%%
sns.distplot(df['fare'],kde=False)

#%%

sns.distplot(df['fare'], kde=False, bins=10)

#%%
sns.jointplot(x='age',y='fare',data=df)

#%%
sns.jointplot(x='age',y='fare',data=df,kind='hex')