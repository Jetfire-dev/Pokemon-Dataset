#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('pokemon_alopez247.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.nunique()


# In[6]:


#cleaning of data
df.isnull().sum()


# In[7]:


df['Type_2']=df['Type_2'].fillna(0)


# In[8]:


df.isnull().sum()


# In[9]:


df[['Pr_Male','Egg_Group_2']]=df[['Pr_Male','Egg_Group_2']].fillna(0)


# In[10]:


#data is cleaned
df.isnull().sum()


# In[11]:


df


# In[12]:


df['Generation']


# In[13]:


x= df[['HP','Attack','Defense','Speed','Sp_Atk','Sp_Def']]


# In[14]:


y= df['hasMegaEvolution']


# In[15]:


x['Overall_Strength'] = x['Attack'] + x['Defense'] + x['Speed'] + x['Sp_Atk'] + x['Sp_Def'] + x['HP']


# In[16]:


# Convert 'hasMegaEvolution' column to numerical values
y1 = y.astype(int)

plt.figure(figsize=(8,5))
plt.scatter(x['Overall_Strength'], y1, c= y1,cmap= 'bwr',marker='o')
plt.xlabel('Overall Strength')
plt.ylabel('Has Mega Evolution')
plt.title('Pokemon Overall Strength and Mega Evolution')

plt.show()


# In[17]:


x.head()


# In[18]:


y1.head()


# In[19]:


from sklearn.model_selection import train_test_split

x_train,x_test,y1_train,y1_test=train_test_split(x,y1,test_size=0.2,random_state=42)


# In[20]:


x_test.shape


# In[21]:


y1_test.shape


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


# Define the class weights
class_weights = {0: 1, 1: 10}  # Assign higher weight to the minority class (1)


# In[24]:


# Create the Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights)

model.fit(x_train,y1_train)


# In[25]:


x_test


# In[26]:


y1_test


# In[27]:


model.predict(x_test)


# In[28]:


model.predict_proba(x_test)


# In[29]:


predictions = model.predict(x_test)


# In[30]:


from sklearn.metrics import precision_score

precision = precision_score(y1_test, predictions)
print("Precision:", precision)


# In[31]:


model.score(x_test,y1_test)


# In[32]:


model.score(x_train,y1_train)


# In[33]:


model.coef_


# In[34]:


model.intercept_


# In[35]:


import math
def sigmoid(x):
    return 1/(1+math.exp(-x))


# In[36]:


def prediction_function(Attack,hasMegaEvolution, coefficients, intercept):
    z = np.sum(Attack * coefficients) + intercept
    y = sigmoid(z)
    
    if hasMegaEvolution == 1:
    
        y *= 1.5
    
    return y


# In[37]:


prediction_function(58,1,model.coef_[0], model.intercept_[0])


# In[ ]:




