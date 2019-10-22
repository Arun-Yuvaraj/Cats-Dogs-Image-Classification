#!/usr/bin/env python
# coding: utf-8

# ## changing directory

# In[1]:


#importing Libraries

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


# In[ ]:


# changing alignment of directories, so that it is used in ImageDataGenerator
dataset_home = 'dataset_dogs_vs_cats/' 
subdirs = ['train/', 'test/']
for subdir in subdirs: 
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/'] 
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True) 
# seed random number generator
seed(1) 
# define ratio of pictures to use for validation 
val_ratio = 0.25 
# copy training dataset images into subdirectories 
src_directory = 'train/train/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio: 
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst) 
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)


# In[21]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1./255)


# In[22]:


train = train_datagen.flow_from_directory('C:/Users/Arun/dataset_dogs_vs_cats/train',target_size=(200,200),class_mode='binary',batch_size=  64)
validation = validation_datagen.flow_from_directory('C:/Users/Arun/dataset_dogs_vs_cats/test',target_size=(200,200),class_mode='binary',batch_size = 64)


# In[24]:



def vgg_block(layer_in,f1,f2,f3):
    layer_in = Conv2D(f1,(3,3),padding ='same',activation ='relu',kernel_initializer='he_uniform')(layer_in)
    layer_in = MaxPooling2D((2,2))(layer_in)
    layer_in = Dropout(0.2)(layer_in)
    layer_in = Conv2D(f2,(3,3),padding ='same',activation ='relu',kernel_initializer='he_uniform')(layer_in)        
    layer_in = MaxPooling2D((2,2))(layer_in)
    layer_in = Dropout(0.2)(layer_in)
    layer_in = Conv2D(f3,(3,3),padding ='same',activation ='relu',kernel_initializer='he_uniform')(layer_in)
    layer_in = MaxPooling2D((2,2))(layer_in)
    layer_in = Dropout(0.2)(layer_in)
    
    return layer_in


# In[25]:


input = Input(shape= (200,200,3))
layer1= vgg_block(input,32,64,128)


# In[31]:


flat = Flatten()(layer1)
dense1 = Dense(128,activation='relu',kernel_initializer='he_uniform')(flat)
drop = Dropout(0.5)(dense1)
output = Dense(1,activation='sigmoid')(drop)


# In[32]:


model = Model(input,output)


# In[33]:


model.summary()


# In[34]:


from keras.optimizers import SGD
opt = SGD(lr = 0.001,momentum = 0.9)
model.compile(optimizer= opt,loss='binary_crossentropy',metrics=['accuracy'])


# In[35]:


history = model.fit_generator(train,steps_per_epoch=len(train),epochs=20)


# In[36]:


acc = model.evaluate_generator(validation, steps=len(validation))

