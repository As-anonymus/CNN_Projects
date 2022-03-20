#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


x_train.shape


# In[6]:


plt.imshow(x_train[0])


# In[7]:


x_train.max()


# In[8]:


x_train = x_train/255


# In[9]:


x_test = x_test/255


# In[10]:


x_train = x_train.reshape(60000,28,28,1)


# In[12]:


x_test = x_test.reshape(10000,28,28,1)


# In[13]:


from tensorflow.keras.utils import to_categorical


# In[14]:


y_cat_train = to_categorical(y_train,10)


# In[15]:


y_cat_test = to_categorical(y_test,10)


# In[16]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[20]:


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[21]:


model.summary()


# In[22]:


model.fit(x_train,y_cat_train,validation_data=(x_test,y_cat_test),epochs=10)


# In[23]:


model.metrics_names


# In[25]:


metrics = pd.DataFrame(model.history.history)


# In[26]:


metrics[['loss','val_loss']].plot()


# In[27]:


metrics[['accuracy','val_accuracy']].plot()


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[29]:


predictions = np.argmax(model.predict(x_test), axis=-1)


# In[30]:


print(classification_report(y_test,predictions))


# In[ ]:




