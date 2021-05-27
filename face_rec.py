
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 06:41:06 2020

@author: User
"""

import numpy as np
import cv2
from random import randrange
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image 


face_cascade = cv2.CascadeClassifier('D:/Desktop/ThesisFolder/Thesis/Cascade/haarcascade_frontalface_default.xml')         #our cascade classifier
model=load_model('D:/Desktop/ThesisFolder/h5files/Emotion_mobilenet.h5')

class_labels=['angry','happy','neutral','sad','surprise']

cap = cv2.VideoCapture(0)       #0 means its default webcam

while True:
    successful_frame_read, frame = cap.read()       #read the video stream
    frame = cv2.flip(frame,1)

    #grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)      #converts to gray color

    face_coordinates = face_cascade.detectMultiScale(frame,1.3,5)    #detect faces


#draw rectangle around face
    for (x,y,w,h) in face_coordinates :
        cv2.rectangle(frame,(x,y),(x+w,y+h),((0),(0),(132)),2) #draws the rectangle with different colors
        roi_gray=frame[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(120,120),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                  
    cv2.imshow('just tryin',frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:         #wait for the Q to stop
        break


cap.release()   #release the video capture
cv2.destroyAllWindows()

