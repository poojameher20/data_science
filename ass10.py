#%%
import pandas as pd
import numpy as np
#%%
#df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\Social_Network_Ads.csv")
#print(df)

#%%
df=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\data-science_pratice\\iris.csv")
print(df)

#%%
print(df.head(5))

#%%
column=len(list(df))
print(column)

#%%
#clearly datset has 5 column indicating 5 features about data
df.info()

#%%
np.unique(df['species'])

#%%
df.describe()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

#%%
fig,axes=plt.subplots(2,2,figsize=(16,8))

axes[0,0].set_title("Distribution of sepal_length")
axes[0,0].hist(df['sepal_length'])

axes[0,1].set_title("Distribution of sepal_width")
axes[0,1].hist(df['sepal_width'])

axes[1,0].set_title("distribution of petal_length")
axes[1,0].hist(df['petal_length'])

axes[1,1].set_title("distribution of petal_width")
axes[1,1].hist(df['petal_width'])


#%%
data_to_plot=[df['sepal_length'],df['sepal_width'],df['petal_width'],df['petal_length']]

#creating a figure instance
fig=plt.figure(1,figsize=(12,8))

#creating an axes instance
ax=fig.add_subplot(111)

#creating the boxplot
bp=ax.boxplot(data_to_plot);




