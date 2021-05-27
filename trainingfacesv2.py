# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:13:27 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:56:04 2020

@author: User
"""
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import os 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt

BATCH_SIZE = 45
Classes = ["angry","happy","neutral","sad","surprise"]

def data_augmentation():
    train_data_dir = "D:/Desktop/ThesisFolder/ThesisFinal2/FinalDataset/train"
    #validation_data_dir='D:/Desktop/ThesisFolder/Thesis/Images/validation'
   
    train_datagen= ImageDataGenerator(
        rescale=1./255,   
        validation_split=0.20,
        zoom_range=0.3,
        width_shift_range =0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        vertical_flip =True,
        fill_mode='nearest')      
    
    validation_gen =ImageDataGenerator(rescale=1./255,validation_split=0.20)
    
    train_generator=train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(48,48),
        batch_size=BATCH_SIZE,
        class_mode='categorical',shuffle=True,subset = 'training')

    validation_generator=validation_gen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(48,48),
        batch_size=BATCH_SIZE,
        class_mode='categorical',shuffle=True,subset='validation')
    
    print(len(train_generator))
    print(len(validation_generator))
    
    imgs, labels = next(iter(train_generator))

    return train_generator,validation_generator


def plots(ims, figsize=(20,10), rows=4, interp=False, titles=None):
     if type(ims[0]) is np.ndarray:
         ims= np.array(ims).astype(np.uint8)
         if (ims.shape[-1] != 3):
             ims = ims.transpose((0,2,3,1))
     f = plt.figure(figsize=figsize)
     cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
     for i in range(len(ims)):
         sp = f.add_subplot(rows, cols, i+1)
         sp.axis('off')
         if titles is not None:
             sp.set_title(titles[i], fontsize=12)
         plt.imshow(ims[i], interpolation=None if interp else 'none')


def building_model():
    model = tf.keras.Sequential()
    
    #layer1 
    model.add(
        Conv2D(64,(3,3),
        padding='same',
        input_shape=(48,48,1),
        activation='relu'))         
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    #layer2 
    model.add(
        Conv2D(128,(3,3),
        padding='same',
        activation='relu'))       
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    
    #layer3
    model.add(
        Conv2D(256,(3,3),
        padding='same',
        kernel_initializer='he_normal',
        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    #layer4
    model.add(
        Conv2D(512,(3,3),
        padding='same',
        kernel_initializer='he_normal',
        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
   
    model.add(Dense(5,activation='softmax'))
    
    print(model.summary())
    
    return model


def create_checkpoint():
    checkpoint = ModelCheckpoint('D:/Desktop/Thesis/Images/Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode ='min',
                             save_best_only=True,
                             verbose=1)
    return checkpoint

def model_fit():
    train_generator,validation_generator = data_augmentation()
    checkpoint = create_checkpoint()
    model = building_model()
    
    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=3,
        verbose=1,
        restore_best_weights=True
                              )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_delta=0.0001)
    
    callbacks = [earlystop,checkpoint,reduce_lr]
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = Adam(lr=0.001),
                  metrics=['accuracy'],)
    
    nb_train_samples = 28820
    #nb_validation_samples = 5937
    epochs=25
    
    history=model.fit_generator(
                    train_generator,
                    steps_per_epoch=510,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=120)


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



































