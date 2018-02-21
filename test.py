# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:41:39 2017

@author: Falah
"""

####just copy and paste the below given code to your shell

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
###### RGB
#path1 = 'dest_colour'    #path of folder of images_colour    
path2 = 'c:\\1\\cnn\\colour'  #path of folder to save images    
#########

########Depth
#path3 = 'dest_depth'    #path of folder of images_depth    
path4 = 'C:\\1\\cnn\\depth'  #path of folder to save images    
############
#listing = os.listdir(path1) 
#num_samples=size(listing)

##### this use oncee to convert image from RGB to grey ##########
#for file in listing:
#    im = Image.open(path1 + '\\' + file)   
##    img = im.resize((img_rows,img_cols))
#    gray = im.convert('L')
#                #need to do some more processing here           
#    gray.save(path2 +'\\' +  file, "JPEG")
#
#### for depth
#for file in listing:
#    im = Image.open(path3 + '\\' + file)   
##    img = im.resize((img_rows,img_cols))
#    gray = im.convert('L')
#                #need to do some more processing here           
#    gray.save(path4 +'\\' +  file, "JPEG")
#
#################################################################


imlist = os.listdir(path2)
num_samples=size(imlist)
print (num_samples)
# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+'\\' + gg)).flatten() for gg in imlist],'f')
# creat label to store class of each image 
label=np.ones((num_samples,),dtype = int)
i=0
k=0
j=1

for file in imlist:
    h=file.split("_")
    class_h=h[0]
    
    if i==0:
        temp=class_h
        print ('ok')
    
    
    if temp!=class_h:
        
        k=k+1
        temp=class_h
    label[i]=k
    i=i+1
    
label=to_categorical(label,num_classes=None)

 #####################################   
imlistd = os.listdir(path4)
num_samples=size(imlistd)

# create matrix to store all flattened images
immatrix_d = array([array(Image.open(path4+'\\' + gg)).flatten() for gg in imlistd],'f')

###################

data1,Label = shuffle(immatrix,label, random_state=2)
train_data1 = [data1,Label]           #variable to data 

data2,Label = shuffle(immatrix_d,label, random_state=2)
train_data2 = [data2,Label]      
#
#print (train_data1[0].shape)
#print (train_data1[1].shape)
#
#print (train_data2[0].shape)
#print (train_data2[1].shape)

(X, y) = (train_data1[0],train_data1[1])
(Z,y)= (train_data2[0],train_data2[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 58, 128, 1)
X_test = X_test.reshape(X_test.shape[0], 58, 128, 1)

Z_train = Z_train.reshape(Z_train.shape[0], 58, 128,1)
Z_test = Z_test.reshape(Z_test.shape[0], 58, 128,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Z_train = Z_train.astype('float32')
Z_test = Z_test.astype('float32')

X_train /= 255
X_test /= 255

Z_train /= 255
Z_test /= 255
#
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
#print('Z_train shape:', Z_train.shape)
#print(Z_train.shape[0], 'train samples')
#print(Z_test.shape[0], 'test samples')
#
#print (y_train.shape)
## convert class vectors to binary class matrices
#y_train = np_utils.to_categorical(y_train, 3)
#y_test = np_utils.to_categorical(y_test, 3)




#%%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Merge
#################model
from keras import optimizers
from keras.callbacks import TensorBoard

#pth.add(MaxPooling2D(pool_size = (4, 4)))
## second layer
depth=Sequential()
depth.add(Convolution2D(32, 3, 3, input_shape= (58, 128,1), activation='relu'))
depth.add(MaxPooling2D(pool_size = (2, 2)))
depth.add(Convolution2D(32, 3, 3, activation='relu'))
depth.add(MaxPooling2D(pool_size = (2, 2)))
depth.add(Convolution2D(64, 3, 3, activation='relu'))
depth.add(MaxPooling2D(pool_size = (2, 2)))

depth.add(Flatten())

#depth.add(Dense(activation = 'softmax', units = 3))
# this for my work classifier.add(Dense(activation = 'softmax', units = 10))
# this for my work classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#classifier.add(Dense(activation = 'sigmoid', units = 1))
#depth.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#####
depth.summary()

colour=Sequential()
colour.add(Convolution2D(32, 3, 3, input_shape= (58, 128,1), activation='relu'))
colour.add(MaxPooling2D(pool_size = (2, 2)))
# second layer
colour.add(Convolution2D(32, 3, 3, activation='relu'))
colour.add(MaxPooling2D(pool_size = (2, 2)))
colour.add(Convolution2D(64, 3, 3, activation='relu'))
colour.add(MaxPooling2D(pool_size = (2, 2)))

colour.add(Flatten())
colour.summary()


###
#classifier=Sequential()
#classifier.add(Merge([depth,colour]))

# this for my work classifier.add(Dense(activation = 'softmax', units = 10))
# this for my work classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#classifier.add(Dense(activation = 'sigmoid', units = 1))
print (X_train.shape)
###############model
mereged=Merge([depth,colour],mode='concat')

classifier=Sequential()
classifier.add(mereged)
classifier.add(Dense(activation='sigmoid',units=128))
classifier.add(Dropout(0.25))

classifier.add(Dense(activation='sigmoid',units=128))
classifier.add(Dropout(0.25))

tensorboard = TensorBoard(log_dir='./logs2', histogram_freq=0,
                          write_graph=True, write_images=False)
classifier.add(Dense(activation = 'softmax', units = 48 ))
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000002)
classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
classifier.summary()

hist = classifier.fit([X_train,Z_train], y_train, batch_size=298, callbacks=[tensorboard], nb_epoch=50,
              verbose=1, validation_data=([X_test,Z_test], y_test))
print(hist.history.keys())
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('accuracy' )
plt.ylabel('accuracy')
plt.xlable('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss' )
plt.ylabel('loss')
plt.xlable('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
