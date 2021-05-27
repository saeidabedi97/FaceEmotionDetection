# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:21:20 2021

@author: Saeid Abedi
"""
#from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
import numpy as np
import os 
import matplotlib.pyplot as plt
from tensorflow.keras import layers
#from tensorflow.keras.layers import BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras import models
from kerastuner.tuners import RandomSearch,BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
#import talos
from keras.optimizers import Adamax
import time

Log_dir =f"{int(time.time())}"

BATCH_SIZE = 32

train_dir = "D:/Desktop/ThesisFolder/ThesisFinal2/FinalDataset/train"
 
validation_dir = "D:/Desktop/ThesisFinal2/DataSet/validation"
 
datagen_train = ImageDataGenerator(
     rescale=1./255,
     validation_split=0.15,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True)

datagen_val = ImageDataGenerator(rescale = 1./255,validation_split=0.15)
 
train_generator = datagen_train.flow_from_directory(
     train_dir,
     #seed=42,
    target_size=(120,120),
     batch_size=BATCH_SIZE,
     shuffle=True,
     subset='training',
     class_mode = 'categorical')
 
 
 #train_augmentation = ImageDataGenerator()
val_generator = datagen_val.flow_from_directory(
     train_dir,
     #seed=42,     
     target_size=(120,120),
     batch_size=BATCH_SIZE, 
     shuffle=True,
     subset='validation',
     class_mode = 'categorical')
 

print(len(train_generator))
print(len(val_generator))


imgs, labels = next(iter(train_generator))

print("Loading the VGG")
base_model = VGG16(input_shape=(120,120,3),
                                               include_top=(False),
                                               pooling=(max),
                                              weights='imagenet')


base_model.summary()
for layer in base_model.layers[:-14]:
    layer.trainable = False

def build_model(hp):
    model = keras.models.Sequential()
    model.add(base_model)
# =============================================================================
#     model.add(layers.Conv2D(hp.Int("input_units",min_value = 128,max_value=680,step=32),
#                      (3, 3), input_shape=(120,120,3),padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     
#     for i in range(hp.Int("n layers",1,4)):
#         model.add(layers.Conv2D(hp.Int(f"conv {i} units",min_value = 32,max_value=286,step=32), (3, 3),padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# =============================================================================

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    for j in range(hp.Int("n layers",1,2)):
                   model.add(layers.Dense(hp.Int(f"Dense {j} units",min_value = 128,max_value = 840,step=32)))
                   model.add(Activation('relu'))
                   

    model.add(layers.Dense(5, activation='softmax'))
    
    model.summary()
    
    model.compile(optimizer=Adamax(
                hp.Choice('learning_rate',
                          values=[1e-3, 1e-4])),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

    return model

tuner = BayesianOptimization(build_model,objective = "val_accuracy",
                     max_trials=6,
                     executions_per_trial = 1,
                     directory = Log_dir)

tuner.search(train_generator,epochs = 8,batch_size = BATCH_SIZE,validation_data=val_generator
             ,steps_per_epoch = 450 ,validation_steps=len(val_generator)//BATCH_SIZE)