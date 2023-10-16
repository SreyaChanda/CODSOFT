#!/usr/bin/env python
# coding: utf-8

# # Importing The Dependencies

# In[20]:


import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[21]:


#loading dataset to a pandas dataframe
credit_card_data = pd.read_csv('C://Users//Sreya//Downloads//creditcard.csv.zip')


# In[22]:


#first 5 rows of the dataset
credit_card_data.head()


# In[23]:


#last 5 rows of the dataset
credit_card_data.tail()


# In[24]:


#dataset information
credit_card_data.info()


# In[25]:


#checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[35]:


#distribution of legit transactions & fradulant transactions
credit_card_data['Class'].value_counts()


# **This Dataset is Highly Unbalanced**

# 0---> Normal Transaction
# 
# 
# 1---> Fraudulent Transaction

# In[36]:


#seperating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[43]:


print(legit.shape)
print(fraud.shape)


# In[44]:


#statistical measures of the data
legit.Amount.describe()


# In[46]:


fraud.Amount.describe()


# In[47]:


#Compare the values for both transactions
credit_card_data.groupby('Class').mean()


#  **Under-Sampling**

# # **Build A Simple Dataset Containing Similar Distribution Of Normal Transaction And Fraudulent Transaction**

# Number of Fraudulant Transactions --> 492

# In[51]:


legit_sample = legit.sample(n=492)


# **Containing two Dataframes**

# In[53]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)


# In[54]:


new_dataset.head()


# In[55]:


new_dataset.tail()


# In[56]:


new_dataset['Class'].value_counts()


# In[58]:


new_dataset.groupby('Class').mean()


# # Splitting The Data Into Features & Targets

# In[59]:


X =new_dataset.drop(columns='Class',axis=1) 
Y =new_dataset['Class']


# In[60]:


print(X)


# In[61]:


print(Y)


# # Split The Data Into Training Data & Testing Data 

# In[62]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[63]:


print(X.shape,X_train.shape,X_test.shape)


# # Model Training 

# **LOGISTIC REGRESSION MODEL**

# In[64]:


model = LogisticRegression()


# In[65]:


#training The LogisticRegression Model With Training Data
model.fit(X_train,Y_train)


# # Model Evaluation

# **Accuracy Score**

# In[71]:


#Accuracy on taining data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[72]:


print('Accuracy on training data:',training_data_accuracy)


# In[74]:


#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[75]:


print('Accuracy on test data:',test_data_accuracy)


# In[ ]:




