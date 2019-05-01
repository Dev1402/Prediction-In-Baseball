#!/usr/bin/env python
# coding: utf-8

# In[143]:


# Import modules
# Data preprocessing

import sqlite3
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#import scikitplot as skplt

# Machine learning
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MSE


# Sean Lahman compiled this data on his website and it was transformed to a sqlite database.

# In[42]:


data = sqlite3.connect('/Users/devanshchawla/Downloads/lahman2016.sqlite')


# In[43]:


# Querying Database for all seasons where a team played 150 or more games and is still active today. 

query = '''select * from Teams inner join TeamsFranchises
on Teams.franchID == TeamsFranchises.franchID
where Teams.G >= 150 and TeamsFranchises.active == 'Y';
'''


# In[44]:


df_1 = data.execute(query).fetchall()
df = pd.DataFrame(df_1)


# In[45]:


df.head()


# In[47]:


df.columns=cols = ['yearID','lgID','teamID','franchID','divID','Rank','G','Ghome','W','L','DivWin','WCWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']


# In[48]:


df.columns


# In[49]:


#dropping uneccessary columns 
col = ['lgID','CS','HBP','franchID','divID','Rank','Ghome','L','DivWin','WCWin','LgWin','WSWin','SF','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']

df = df.drop(col, axis=1)
df.head()


# In[50]:


#checking out the null values
df.info()


# In[67]:


# DP, SO Hhave problem with missing values
#checking the feature variance
print(df['yearID'].nunique())
df.describe()


# In[54]:


# For DP And SO we will have mean
df['DP'] = df['DP'].fillna(df['DP'].mean())
df['SO'] = df['SO'].fillna(df['SO'].mean())
df.info()


# In[57]:


plt.hist(df['W'])


# In[61]:


#Average win per year
win = np.mean(df['W'])
win


# In[84]:


#filtering the data for >1900
df = df[df['yearID']>1900]
df = df.sort_index(by='yearID')


# In[85]:


#Now creating new variables
runs_per_year ={}
games_per_year = {}

for i, row in df.iterrows():
    runs = row['R']
    year = row['yearID']
    game = row ['G']
    
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year]+runs
        games_per_year[year] =  games_per_year[year]+game
    else:
        runs_per_year[year] = runs
        games_per_year[year] = game

print(runs_per_year)
print('Line break\n')
print(games_per_year)
        


# In[89]:


#runs per game per year
run_game_year = {}

for k,v in games_per_year.items():
    year = k
    game = v
    runs = runs_per_year[k]
    
    run_game_year[year] = runs/game
    
print(run_game_year)    


# In[104]:


def assign_mlb_rpg(year):
    return run_game_year[year]

df['run_perg_pery'] = df['yearID'].apply(assign_mlb_rpg)
df['run_perg_pery'].head()


# In[105]:


# Create new features for Runs per Game and Runs Allowed per Game
df['R_per_game'] = df['R'] / df['G']
df['RA_per_game'] = df['RA'] / df['G']


# In[115]:



plt.subplot(1,2,1)
plt.ylabel('Wins')
plt.scatter(df['R_per_game'], df['W'], c= 'green')
plt.xlabel('Run scored per game')

plt.subplot(1,2,2)
plt.scatter(df['RA_per_game'], df['W'], c= 'orange')
plt.xlabel('Run allowed per game')


# In[127]:


df.columns


# In[129]:


df.head()


# In[137]:


#splitting the variables
X = df.drop(['W', 'teamID'], axis = 1).values
y = df['W'].values


# In[ ]:





# In[140]:


#Splitting the data
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[145]:


lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
mse = MSE(y_test,y_pred)
RMSE = np.sqrt(mse)

print('error :', RMSE)


# In[155]:


#using Ridge regression
from sklearn.linear_model import RidgeCV

x= [0.01,0.1,0.2,0.5,1,2,5,10]

ridge = RidgeCV(alphas = x, normalize= True)

ridge.fit(X_train,y_train)
y_pred= ridge.predict(X_test)
mse = MSE(y_test,y_pred)
RMSE = np.sqrt(mse)

print('error :', RMSE)

