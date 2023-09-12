#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification :
# 
# ## Make a system which tells whether the person will be save from sinking. What factors were most likely lead to success-socio-economic status, age, gender and more.

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import warnings
warnings.simplefilter('ignore')


# In[4]:


df= pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isna().sum(),df_test.isna().sum()


# In[9]:


df.columns,df_test.columns


# In[10]:


df.drop(['Cabin'],axis=1,inplace=True)
df_test.drop(['Cabin'],axis=1,inplace=True)


# In[11]:


# Imoutation techinique
## 1 mean value imoutation 


# # imputation techinique
# ## 1 mean value imoutation 

# In[12]:


# plt.hist(df.Age)
sns.distplot(df.Age)


# In[13]:


sns.distplot(df_test.Age)


# In[14]:


df.Age.isnull().sum()


# In[15]:


df['Age_mean']=df['Age'].fillna(df['Age'].mean())
df_test['Age_mean']=df_test['Age'].fillna(df['Age'].mean())


# In[16]:


df[['Age_mean','Age']]


# In[17]:


import seaborn as sns
sns.distplot(df['Age'])
sns.distplot(df['Age_mean'],color='r')


# In[18]:


sns.distplot(df_test['Age'])
sns.distplot(df_test['Age_mean'],color='r')


# ## 2. Median value imputation
# ### if you have outlieirs in data set
# 

# In[19]:


# df['Age_Median']=df.Age.fillna(df['Age'].median())


# 

# In[20]:


# df[['Age_Median','Age']]


# In[21]:


# import seaborn as sns
# sns.distplot(df['Age'])
# sns.distplot(df['Age_Median'],color='y')


# In[22]:


##


# In[23]:


df.info()
df.drop('Age',axis=1,inplace=True)


# In[24]:


df_test.info()
df_test.drop('Age',axis=1,inplace=True)


# In[25]:


df.info(),df_test.info()


# ## 3. mode value imputation
# 
# ### use in categorical value
# 

# In[26]:


df[df['Embarked'].isnull()]


# In[27]:


df['Embarked'].unique()


# In[28]:


mode=df['Embarked'].mode()[0]

mode


# In[29]:


df['Embarked_mode']=df['Embarked'].fillna(mode)


# In[30]:


df['Embarked_mode'].isnull().sum()


# In[31]:


df.drop(['Embarked','Ticket','Name'],inplace=True,axis=1)


# In[32]:


df.columns


# In[33]:


df.info()


# In[34]:


df.rename({'Embarked_mode':'Embarked','Age_mean':'Age'},inplace=True,axis=1)
df


# In[35]:


df_test.rename({'Age_mean':'Age'},inplace=True,axis=1)


# In[36]:


df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[37]:


df.info()


# In[38]:


df['Embarked'].unique()


# In[39]:


df_test.info()


# In[40]:


df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_test.info()


# In[41]:


sns.distplot(df_test['Fare'])


# In[42]:


import numpy as np
dfFare=np.log(df_test['Fare'])


# In[43]:


FareMode=df_test['Fare'].median()


# In[44]:


df['Fare']=df['Fare'].fillna(FareMode)
df_test['Fare']=df_test['Fare'].fillna(FareMode)


# In[45]:


df_test.info()


# In[46]:


df.columns


# In[47]:


df_test.columns


# In[48]:


x=df[[ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y=df['Survived']


# In[49]:


sns.heatmap(df.corr())


# In[50]:


sns.histplot(data=df,x='Age',
#              y='Survived',
             kde=True,hue='Survived')


# In[51]:


sns.histplot(data=df,x='Pclass',
#              y='Survived',
             kde=True,hue='Survived')


# In[52]:


sns.histplot(data=df,x='Fare',
#              y='Survived',
             kde=True,hue='Survived',bins =20)


# In[53]:


sns.histplot(data=df,x='Sex',
             kde=True,hue='Survived')


# In[54]:


sns.histplot(data=df,x='Embarked',
             y='Survived',
             kde=True,bins=2)


# ### ****We can see that fare, Sex is highly corelated to survived column irrespective of their Age. We Observe that The Increase in Fare is directly relate of survival.we Also see that ratio of survival of Female is more than male.We Also see that Age Group of 20-40 has larger survival rate than other group so our model shoud  work best if we treat Age As Categorical Varible and model should be non linear and Should be able to categorise the data **
# 

# In[ ]:





# In[55]:


# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# In[56]:


from sklearn.linear_model import LogisticRegression


# In[57]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[58]:


y_pred=model.predict(X_val)


# In[59]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_val)


# In[60]:


x_test=df_test[[ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_testPred=model.predict(x_test)


# In[61]:


y_testPred


# In[62]:


submmision=df_test.join(pd.DataFrame(y_testPred))


# In[63]:


submmision.rename({0:'Survived'},axis=1,inplace=True)


# In[64]:


submmision[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Survived']]


# In[65]:


submmision['Sex'] = submmision['Sex'].map({0:'male', 1:'female'})
submmision['Embarked'] = submmision['Embarked'].map({0:'S', 1:'C', 2:'Q'})


# In[66]:


submmision[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Survived']]


# In[ ]:





# In[67]:


df.corr()


# In[ ]:




