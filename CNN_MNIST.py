#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


from tensorflow.keras.datasets import mnist


# In[35]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[36]:


x_train.shape


# In[37]:


single_image=x_train[0]


# In[38]:


single_image.shape


# In[39]:


single_image


# In[40]:


plt.imshow(single_image)


# In[41]:


y_train


# In[42]:


from tensorflow.keras.utils import to_categorical


# In[43]:


y_train.shape


# In[44]:


y_example = to_categorical(y_train)


# In[45]:


y_example.shape


# In[46]:


y_example[0]


# In[47]:


y_cat_test = to_categorical(y_test,num_classes =10)


# In[48]:


y_cat_train = to_categorical(y_train,num_classes =10)


# In[49]:


single_image.max()


# In[50]:


single_image.min()


# In[51]:


x_train = x_train/255


# In[52]:


x_test = x_test/255


# In[53]:


scaled_image = x_train[0]


# In[54]:


scaled_image.max()


# In[55]:


plt.imshow(scaled_image)


# In[56]:


x_train.shape


# In[57]:


x_test.shape


# In[29]:


#batch size,width,height,color channel
x_train = x_train.reshape(60000,28,28,1)


# In[58]:


x_test = x_test.reshape(10000,28,28,1)


# In[59]:


from tensorflow.keras.models import Sequential


# In[60]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[64]:


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
          
model.add(MaxPool2D(pool_size=(2,2)))
          
          
model.add(Flatten())
          
model.add(Dense(128,activation='relu'))
          
#out layer ---> softmax ---> Multiclass
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[65]:


from tensorflow.keras.callbacks import EarlyStopping


# In[66]:


early_stop = EarlyStopping(monitor='val_loss',patience=1)


# In[68]:


model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])


# In[69]:


metrics = pd.DataFrame(model.history.history)


# In[71]:


metrics[['loss','val_loss']].plot()


# In[72]:


metrics[['accuracy','val_accuracy']].plot()


# In[73]:


model.metrics_names


# In[74]:


model.evaluate(x_test,y_cat_test)


# In[75]:


from sklearn.metrics import classification_report,confusion_matrix


# In[77]:


predictions = np.argmax(model.predict(x_test), axis=-1)


# In[78]:


y_cat_test.shape


# In[79]:


y_test


# In[80]:


print(classification_report(y_test,predictions))


# In[81]:


confusion_matrix(y_test,predictions)


# In[85]:


sns.heatmap(confusion_matrix(y_test,predictions),annot=True)


# In[86]:


my_number = x_test[7]


# In[88]:


plt.imshow(my_number.reshape(28,28))


# In[98]:


t = my_number.reshape(1,28,28,1)


# In[101]:


predict_x=model.predict(t) 
classes_x=np.argmax(predict_x,axis=1)


# In[102]:


classes_x


# In[ ]:




