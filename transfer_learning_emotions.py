# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:00:10 2021

@author: User
"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
#from tensorflow import keras
from keras import regularizers
import  keras
import numpy as np
import os 
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import BatchNormalization,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
#import talos
from keras.optimizers import Adamax


BATCH_SIZE = 45
Classes = ["angry","happy","neutral","sad","surprise"]

def Data_augmentation():
    train_data_dir='D:/Desktop/ThesisFolder/ThesisFinal2/FinalDataset/train'
    #validation_data_dir='D:/Desktop/ThesisFolder/Thesis/Images/validation'
    datagen_train = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

    datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)    

    train_generator = datagen_train.flow_from_directory(
    train_data_dir,
    #seed=42,
    target_size=(100,100),
    batch_size=64, 
    shuffle=True,
    subset='training',
    class_mode = 'categorical',interpolation = 'nearest')

    val_generator = datagen_val.flow_from_directory(
    train_data_dir,
    #seed=42,
    target_size=(100,100),
    batch_size=64, 
    shuffle=True,
    subset='validation',
    class_mode = 'categorical',
    interpolation = 'nearest')
   
    print(len(val_generator))
    print(len(train_generator))
    
    imgs, labels = next(iter(train_generator))

    return val_generator,train_generator

# =============================================================================
# def plots(ims, figsize=(20,10), rows=4, interp=False, titles=None):
#     if type(ims[0]) is np.ndarray:
#         ims= np.array(ims).astype(np.uint8)
#         if (ims.shape[-1] != 3):
#             ims = ims.transpose((0,2,3,1))
#     f = plt.figure(figsize=figsize)
#     cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
#     for i in range(len(ims)):
#         sp = f.add_subplot(rows, cols, i+1)
#         sp.axis('off')
#         if titles is not None:
#             sp.set_title(titles[i], fontsize=12)
#         plt.imshow(ims[i], interpolation=None if interp else 'none')
# =============================================================================

def building_model():
    print("Loading the VGG")
    base_model = VGG16(input_shape=(100,100,3),
                           include_top=(False),
                           pooling=(max),
                           weights='imagenet')
    
    
    for layer in base_model.layers:
        layer.trainable = False
        
    base_model.summary()       
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(313,
    	activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, 
    	activation='softmax'))
    
    model.summary()

    return model


def create_checkpoint():
    checkpoint = ModelCheckpoint('D:/Desktop/ThesisFolder/h5files/Emotion_mobilenetnew2.h5',
                             monitor='val_loss',
                             mode ='min',
                             save_best_only=True,
                             verbose=1)
   
    return checkpoint

def model_fit():
    
    epochs=30

    val_generator ,train_generator = Data_augmentation()
    checkpoint = create_checkpoint()                             
    model = building_model()
    
    earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)
    
    callbacks = [checkpoint,reduce_lr]
    model.compile(loss = "categorical_crossentropy",
              optimizer = Adamax(lr = 0.0001),
              metrics=['accuracy'])
    
    with tf.device("/device:GPU:0"):
        history=model.fit_generator(train_generator,
                   steps_per_epoch=520,
                   epochs=epochs,
                   callbacks=callbacks,
                   validation_data=val_generator,
                  validation_steps=len(val_generator)//BATCH_SIZE)
        
    return history
        
def plot_graph():
    history = model_fit()
    print("making the graph...")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    print("Loss")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()    
     
 

if __name__ == '__main__':
    
    model_fit()










