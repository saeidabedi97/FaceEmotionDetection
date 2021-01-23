# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:20:13 2021

@author: User
"""
import tensorflow as tf
#from tensorflow import keras
import keras
import numpy as np
import os 
import matplotlib.pyplot as plt
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import models
#from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dir = 'D:/Desktop/Thesis3/train'
validation_dir = 'D:/Desktop/Thesis3/validation'

BATCH_SIZE = 32
IMG_SIZE = (48,48)

NUM_CLASSES = 7


train_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,      #0-180
                                      width_shift_range=0.2,  #fraction of total w
                                      height_shift_range=0.2, #fraction of total h
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
   
valid_datagen = ImageDataGenerator(rescale=1./255)
    
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                        target_size=(160,160), 
                                                        batch_size=BATCH_SIZE, 
                                                        class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(validation_dir, 
                                                        target_size=(160,160),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')


#plotting the first pictures of each class
#class_names = train_dataset.class_names
#plt.figure(figsize=(10, 10))
#for images, labels in train_dataset.take(1):
 # for i in range(9):
  #  ax = plt.subplot(3, 3, i + 1)
   # plt.imshow(images[i].numpy().astype("uint8"))
    #plt.title(class_names[labels[i]])
    #plt.axis("off")
    
    
#to find out the number of bacthes in test set and generate the test set 
#our dataset has only train and validation 
#we move 20% of validation dataset to test set 
#val_batches = tf.data.experimental.cardinality(validation_dataset)
#test_dataset = validation_dataset.take(val_batches // 5)
#validation_dataset = validation_dataset.skip(val_batches // 5)  

#to print the number of baatches we have in validation and test dataset
#print("Number of validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
#print('Number of test batches: %d '  % tf.data.experimental.cardinality(test_dataset))


#better performance with tf.data method for input piplines to read from disk 
#AUTOTUNE = tf.data.AUTOTUNE
#prefetching datas in our train, validation and test dataset
#train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)          
#validation_dataset = validation_dataset.prefetch(buffer_size = AUTOTUNE)
#test_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
    

 


#this model expects pixel values in [-1,1] 
#but pixel values in our images are [0,255]
#to rescale them we can use the preprocessing method included with the model

preprocess_input = keras.applications.mobilenet_v2.preprocess_input

#create the base model from the pretrained model mobilenetv2
#it converts 48*48*3 image to 2*2*1280 block of features 

IMG_SHAPE = (160,160,3)
base_model = keras.applications.MobileNetV2(input_shape=(IMG_SHAPE),
                                               include_top=(False),
                                              weights='imagenet')

image_batch, label_batch = next(iter(train_generator))
feature_batch = base_model(image_batch)
#print(feature_batch.shape)


#before Feature extraction we have to freeze the conv layers 
#because we need to create our classifire on top of our model 
base_model.trainable = False
#imagenet has many layers but we only need the base model to freeze

#base_model.summary()

#Adding the classification head 
#converting the features to a single 1280 element vector per image 
global_average_layer = keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


#using dense to convert these features into a single prediction per image  
#prediction_layer = layers.Dense(512,activation='relu')
#prediction_batch = prediction_layer(feature_batch_average)
#print(prediction_batch.shape)



model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu',kernel_initializer='he_normal'))
model.add(layers.Dense(256,activation='relu',kernel_initializer='he_normal'))
model.add(layers.Dense(128,activation='relu',kernel_initializer='he_normal'))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
#model.summary()



#compile the model before training 
base_learning_rate = 0.0001
model.compile(optimizer = keras.optimizers.Adam(lr=base_learning_rate),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint = ModelCheckpoint('D:/Desktop/Thesis5/Emotion_mobilenet.h5',
                             monitor='val_loss',
                             mode ='min',
                             save_best_only=True,
                             verbose=1)

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

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 28820
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//BATCH_SIZE,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=valid_generator,
                validation_steps=nb_validation_samples//BATCH_SIZE)












    
    
    
    
    
    
    
    
    
    
    
    
    