#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[4]:


data_dir = 'C:\\Users\\Aditya Singh\\Downloads\\cell_images\\cell_images'


# In[6]:


os.listdir(data_dir)


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


from matplotlib.image import imread


# In[9]:


test_path = data_dir+'\\test\\'
train_path = data_dir+'\\train\\'


# In[10]:


test_path


# In[12]:


os.listdir(test_path)


# In[14]:


os.listdir(train_path)


# In[16]:


os.listdir(train_path+'parasitized')[0]


# In[17]:


para_cell=train_path+'parasitized\\'+'C100P61ThinF_IMG_20150918_144104_cell_162.png'


# In[18]:


para_cell


# In[19]:


imread(para_cell).shape


# In[20]:


plt.imshow(imread(para_cell))


# In[21]:


os.listdir(train_path+'uninfected')[0]


# In[22]:


uninfected_cell=train_path+'uninfected\\'+'C100P61ThinF_IMG_20150918_144104_cell_128.png'


# In[23]:


uninfected_cell


# In[24]:


imread(uninfected_cell).shape


# In[25]:


plt.imshow(imread(uninfected_cell))


# In[26]:


#pwd


# In[27]:


len(os.listdir(train_path+'parasitized'))


# In[28]:


len(os.listdir(train_path+'uninfected'))


# In[29]:


len(os.listdir(test_path+'parasitized'))


# In[30]:


len(os.listdir(test_path+'uninfected'))


# In[31]:


dim1 = []
dim2 = []

for image_filename in os.listdir(test_path+'uninfected'):
    
    img = imread(test_path+'uninfected\\'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
    


# In[33]:


sns.jointplot(dim1,dim2)


# In[34]:


np.mean(dim1)


# In[35]:


np.mean(dim2)


# In[36]:


image_shape = (130,130,3)


# In[37]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[38]:


help(ImageDataGenerator)


# In[46]:


image_gen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')


# In[40]:


#imread(para_cell).max()


# In[44]:


#imread(uninfected_cell).min()


# In[51]:


para_img = imread(para_cell)

plt.imshow(para_img)


# In[53]:


plt.imshow(image_gen.random_transform(para_img))


# In[54]:


train_path


# In[55]:


image_gen.flow_from_directory(train_path)


# In[56]:


#os.listdir(train_path)


# In[57]:


image_gen.flow_from_directory(test_path)


# In[58]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout


# In[59]:


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[61]:


model.summary()


# In[62]:


from tensorflow.keras.callbacks import EarlyStopping


# In[63]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[64]:


batch_size= 16


# In[66]:


train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],color_mode='rgb',batch_size=batch_size,class_mode='binary')


# In[67]:


test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2],color_mode='rgb',batch_size=batch_size,class_mode='binary',shuffle=False)


# In[68]:


train_image_gen.class_indices


# In[69]:


results = model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,callbacks=[early_stop])


# In[70]:


model.history.history


# In[71]:


model.evaluate_generator(test_image_gen)


# In[72]:


model.metrics_names


# In[73]:


pred = model.predict_generator(test_image_gen)


# In[83]:


predi = pred > 0.80


# In[84]:


predi


# In[85]:


len(pred)


# In[86]:


from sklearn.metrics import classification_report,confusion_matrix


# In[87]:


test_image_gen.classes


# In[88]:


print(classification_report(test_image_gen.classes,predi))


# In[89]:


confusion_matrix(test_image_gen.classes,predi)


# In[90]:


para_cell


# In[91]:


from tensorflow.keras.preprocessing import image


# In[95]:


my_img = image.load_img(para_cell,target_size=image_shape)


# In[96]:


my_img


# In[97]:


model.summary()


# In[98]:


my_img_arr = image.img_to_array(my_img)


# In[99]:


my_img_arr


# In[100]:


my_img_arr.shape


# In[101]:


my_img_arr = np.expand_dims(my_img_arr,axis=0)


# In[102]:


my_img_arr.shape


# In[103]:


model.predict(my_img_arr)


# In[104]:


train_image_gen.class_indices


# In[ ]:




