# Disable warning
import warnings
warnings.filterwarnings('ignore')


#OpenCV, Pandas and other essentials
import os
import random
import numpy as np
import pandas as pd 
import cv2 as cv

#For plotting
import matplotlib.pyplot as plt

#Keras 
from keras.utils import load_img

#sklearn train test split
from sklearn.model_selection import train_test_split


#Keras Models Layers and Optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam   
 
#Keras Early Stopping
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Keras Data Preprocessing
from keras.preprocessing.image import ImageDataGenerator

#Keras Saving Model
from keras.models import load_model
from keras.models import model_from_json

#Keras Metrics
from sklearn.metrics import roc_curve, auc

np.random.seed(1984)

#First I iterate through my folder that has the data I manually classified into ball or not ball

## -- VARIABLE DEFINITION

#  declare the location of our training and validation files
base_dir = '../volleyball-tracking/training/basedir'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# cat and dog folders for training
train_cats_dir = os.path.join(train_dir, 'ball')
train_dogs_dir = os.path.join(train_dir, 'notball')

# cat and dog folders for validation
validation_cats_dir = os.path.join(validation_dir, 'ball')
validation_dogs_dir = os.path.join(validation_dir, 'notball')


#Define constants

FAST_RUN = False
IMAGE_WIDTH=32
IMAGE_HEIGHT=32
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
BATCH_SIZE = 32

#model architecture

model = Sequential([
    
Conv2D(32,(3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
MaxPooling2D(),

Conv2D(64,(3,3), activation='relu'),
MaxPooling2D(),

Flatten(),
Dense(64, activation='relu'),
Dropout(0.1),
Dense(1, activation='sigmoid'),
])

model.summary()

opt = RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])


earlystop = EarlyStopping(patience=10)

#Callbacks

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

#preprocess data

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

test_datagen  = ImageDataGenerator(rescale = 1.0/255)


#Training generator

batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(32, 32))


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=batch_size,
                                                         class_mode='binary',
                                                         target_size=(32, 32))


#fit model w data
steps_per_epoch = len(train_generator)//batch_size

validation_steps = len(validation_generator)//batch_size # if you have validation data 

history = model.fit(
            train_generator, # pass in the training generator
            steps_per_epoch=steps_per_epoch,
            epochs=50,
            validation_data=validation_generator, # pass in the validation generator
            validation_steps=validation_steps//BATCH_SIZE,
            )

model_json = model.to_json()
with open("./models/newmodel.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./models/newmodel.h5")
