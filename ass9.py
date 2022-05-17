#%%
import numpy as np
import seaborn as sns

#%%
df=sns.load_dataset('titanic')

#%%
df.head()

#%%
df.describe()

#%%
df.isnull().sum()

#%%
df['age'][:20]

#%%
df['age']=df['age'].replace(np.NAN,df['age'].mean())
df.isnull().sum()
#%%
sns.boxplot(x='sex',y='age',data=df)

#%%
sns.boxplot(x='sex',y='age',data=df,hue='survived')