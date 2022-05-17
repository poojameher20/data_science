#%%
import pandas as pd
import numpy as np

#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\mall.csv")
print(df)

#%%
print(df.head(5))

#%%
print(df.describe())

#%%
print(df.isnull().sum())


#%%
print(df.dtypes)
'''#%%
df['age'][:20]

df['age']=df['age'].replace(np.NAN,df['age'].mean())
print(df.isnull().sum())
'''


#%%
import seaborn as sns
sns.boxplot(df['Age'])
#%%

#to find outliers
df=df[~(df['Age']<=30)]

#%%
#data transfomation
import seaborn as sns

sns.distplot(df['reading score'])

#%%
#data transform normalization
from scipy import stats

nordata=stats.boxcox(df['reading score'])

sns.histplot(nordata[0])
