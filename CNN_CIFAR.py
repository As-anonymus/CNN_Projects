#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from tensorflow.keras.datasets import cifar10


# In[3]:


(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[4]:


x_train.shape


# In[6]:


x_train[0].shape


# In[9]:


#plt.imshow(x_train[89])


# In[10]:


x_train[0].max()


# In[11]:


x_train = x_train/255


# In[12]:


x_test = x_test/255


# In[13]:


x_test.shape


# In[15]:


y_test


# In[16]:


from tensorflow.keras.utils import to_categorical


# In[17]:


y_cat_train = to_categorical(y_train,10)


# In[18]:


y_cat_test = to_categorical(y_test,10)


# In[19]:


y_train[0]


# In[20]:


plt.imshow(x_train[0])


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[22]:


model = Sequential()

#convolution layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
#Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))


#convolution layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
#Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
          
model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[23]:


model.summary()


# In[24]:


from tensorflow.keras.callbacks import EarlyStopping


# In[29]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[26]:


model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])


# In[30]:


metrics = pd.DataFrame(model.history.history)


# In[31]:


metrics.head()


# In[32]:


metrics.columns


# In[33]:


metrics[['accuracy','val_accuracy']].plot()


# In[34]:


metrics[['loss', 'val_loss']].plot()


# In[35]:


model.evaluate(x_test,y_cat_test,verbose=0)


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix


# In[37]:


predictions = np.argmax(model.predict(x_test), axis=-1)


# In[38]:


print(classification_report(y_test,predictions))


# In[39]:


sns.heatmap(confusion_matrix(y_test,predictions),annot=True)


# In[47]:


my_image = x_test[16]


# In[48]:


plt.imshow(my_image)


# In[49]:


y_test[16]


# In[44]:


t = my_image.reshape(1,32,32,3)


# In[45]:


predict_x=model.predict(t) 
classes_x=np.argmax(predict_x,axis=1)


# In[46]:


classes_x


# In[ ]:




