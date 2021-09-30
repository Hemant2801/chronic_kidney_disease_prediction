#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras as k
from keras.models import Sequential,load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt


# In[2]:


#loading data

data = pd.read_csv('kidney_disease.csv')
data.columns


# We use the following representation to collect the dataset
# age - age
# bp - blood pressure
# sg - specific gravity
# al - albumin
# su - sugar
# rbc - red blood cells
# pc - pus cell
# pcc - pus cell clumps
# ba - bacteria
# bgr - blood glucose random
# bu - blood urea
# sc - serum creatinine
# sod - sodium
# pot - potassium
# hemo - hemoglobin
# pcv - packed cell volume
# wc - white blood cell count
# rc - red blood cell count
# htn - hypertension
# dm - diabetes mellitus
# cad - coronary artery disease
# appet - appetite
# pe - pedal edema
# ane - anemia
# class - class

# In[3]:


#shape of data
data.shape


# In[4]:


data.head()


# In[5]:


#list of column name
data_column = ['sg', 'al' , 'sc' , 'hemo' , 'pcv' , 'wc' , 'rc' , 'htn' , 'classification']

#taking only features present in data_colimn
df = data.drop([column for column in data.columns if not column in data_column] , axis=1)
    
df.head()


# In[6]:


#calculating nan values
df.isna().sum()

#removing nan values
df = df.dropna(axis=0)
#df.isna().sum()


# In[7]:


#trsnform non numeric data
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[8]:


df.head()


# In[9]:


#splitting the data foe testing and training
x = df.drop(['classification'] ,axis= 1)
y = df['classification']


# In[10]:


#scaling
#using min-max-scaler this converts the data in to 0 to 1
x_scaler = MinMaxScaler()
x_scaler.fit(x)   #fit the data for the x dataframe
column_name = x.columns #stores the columns name
x[column_name] = x_scaler.transform(x) #will transform the x dataform across the columns


# In[11]:


#splitting the data in train and test set
x_train  ,x_test,y_train ,y_test = train_test_split(x , y ,test_size=0.2 ,shuffle=True )


# In[12]:


#building the model
#Sequential groups a linear stack of layers into a tf.keras.Model.
#Sequential provides training and inference features on this model.
#Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer
#Initializers define the way to set the initial random weights of Keras layers.

model = Sequential()
model.add(Dense(256 ,input_dim=len(x.columns), kernel_initializer = k.initializers.random_normal(seed =13), activation = 'relu'))
model.add(Dense(1,activation= 'hard_sigmoid'))


# In[13]:


#comppile the model
#loss function are of mant types
#for more info :- https://data-flair.training/blogs/compile-evaluate-predict-model-in-keras/

model.compile(loss= 'binary_crossentropy' , optimizer = 'adam' , metrics=['accuracy'])


# In[14]:


#train the model
training = model.fit(x_train,y_train , epochs=2000 ,batch_size = x_train.shape[0])


# In[15]:


#saving the model
model.save('chronic_kidney_disease.model')


# In[16]:


#visualizing the model in loss and accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['loss'])
plt.title('MODEL ACCURACY AND LOSS')
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY AND LOSS')
plt.show()


# In[17]:


print(f'training data shape :{x_train.shape}')
print(f'training data shape :{x_test.shape}')


# In[18]:


pred = model.predict(x_test)
pred = [1 if y>0.5 else 0 for y in pred]
pred


# In[19]:


y_test = [i for i in y_test]
print(f'ACTUAL VALUES: {y_test}')
print(f'PREDICTED VALUE: {pred}')


# In[ ]:




