#!/usr/bin/env python
# coding: utf-8

# **Customer Churn Prediction - Machine Learning Intern Assessment**

# **Step 1: Import Necessary Libraries and Load the Dataset**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV


# In[2]:


# Load the dataset
data = pd.read_excel(r"C:\Users\asmit\OneDrive\Desktop\customer_churn_large_dataset.xlsx")


# **EDA**

# In[3]:


# Display the first few rows of the dataset to understand its structure
data.head()


# **Step 2: Data Preprocessing**

# In[4]:


# Display basic information about the dataset
print("Dataset Information:")
print(data.info())


# In[5]:


# Display summary statistics of numeric columns
print("\nSummary Statistics:")
print(data.describe())


# In[6]:


# Check the first few rows of the dataset
print("\nFirst Few Rows:")
print(data.head())


# In[7]:


# Check the shape of the dataset (number of rows and columns)
print("\nDataset Shape:")
print(data.shape)


# In[8]:


# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())


# In[9]:


data.corr()


# In[10]:


print("\nUnique Values in Categorical Columns:")
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    unique_values = data[column].unique()
    print(f"{column}: {unique_values}")

# Check class distribution for the target variable (Churn)
print("\nClass Distribution for Churn:")
print(data['Churn'].value_counts())


# In[11]:


churn1 = data.loc[data["Churn"]==1]


# In[12]:


not_churn = data.loc[data["Churn"]==0]


# In[13]:


data.groupby("Churn").agg("mean")


# In[14]:


data.groupby("Gender").agg({"Age": "mean"})


# In[15]:


data.groupby("Gender").agg({"Churn": "mean"})


# In[ ]:





# In[16]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Churn'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('dağılım')
ax[0].set_ylabel('')
sns.countplot('Churn',data=data,ax=ax[1])
ax[1].set_title('Churn')
plt.show()


# In[ ]:





# In[17]:


#Handling Categorical Variable
# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Location'], drop_first=True)


# In[18]:


#Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and testing sets (e.g., 80% train, 20% te  #train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Step 3: Feature Engineering**

# In[19]:


df_train = data.sample(frac=0.8,random_state=200)
df_test = data.drop(df_train.index)
print(len(df_train))
print(len(df_test))

