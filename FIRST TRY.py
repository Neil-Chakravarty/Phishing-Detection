#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("C:/Users/chakr/Downloads/dataset_phishing.csv")
df.head()


# In[4]:


df.describe()


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


df.isnull().sum().sum()


# In[7]:


df["status"].value_counts()


# In[8]:


df["status"] = df["status"].map({"legitimate":0, "phishing":1})


# In[9]:


df["status"].value_counts()


# In[10]:


df.info()


# In[11]:


df.corr()


# In[12]:


x = df.iloc[:,1:-1]
y = df["status"]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 42, test_size = 0.3)


# In[13]:


df.info()


# In[14]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.score(x_train, y_train)


# In[15]:


y_predict = rfc.predict(x_test)
print(y_predict)


# In[16]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_predict))


# In[17]:


print(classification_report(y_test, y_predict))


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize= (100,100))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize= (100,100))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[ ]:




