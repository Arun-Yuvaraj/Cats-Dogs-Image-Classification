#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import libraries
from os import makedirs 
from os import listdir 
from shutil import copyfile 
from random import seed 
from random import random
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam


# In[4]:


# taking VGG16 model,removing the classifier part and training classifier according to our data
def model():
    model = VGG16(include_top = False,input_shape = (224,224,3))
    for layers in model.layers:
        layers.trainable = False
    flat = Flatten()(model.layers[-1].output)
    dense1 = Dense(512,activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(512,activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)
    output = Dense(1,activation='sigmoid')(drop2)
    model = Model(model.inputs,output)
    return model


# In[5]:


model = model()
model.compile(optimizer = Adam(lr = 0.0001),loss = 'binary_crossentropy',metrics = ['accuracy'])


# In[6]:


model.summary()


# In[7]:


train_datagen = ImageDataGenerator(
    
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center = True
)

train_datagen.mean = [123.68, 116.779, 103.939]

validation_datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)
validation_datagen.mean = [123.68, 116.779, 103.939]


# In[8]:


train = train_datagen.flow_from_directory('C:/Users/Arun/dataset_dogs_vs_cats/train',target_size=(224,224),class_mode='binary',batch_size=  64)
validation = validation_datagen.flow_from_directory('C:/Users/Arun/dataset_dogs_vs_cats/test',target_size=(224,224),class_mode='binary',batch_size = 64)


# In[9]:


history = model.fit_generator(train,steps_per_epoch=len(train),epochs=50)


# In[ ]:


acc = model.evaluate_generator(validation, steps=len(validation))


# In[1]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[2]:


import tensorflow as tf
tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )


# In[ ]:




