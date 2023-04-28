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
    
    Conv2D(16,(3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    Dense(1, activation='sigmoid')  
])

model.summary()
model.compile(loss='binary_crossentropy', optimizer="adam", metrics = ['accuracy'])


earlystop = EarlyStopping(patience=10)

#Callbacks

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)

# callbacks = [earlystop, learning_rate_reduction]

#preprocess data

train_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen  = ImageDataGenerator(rescale = 1.0/255)


#Training generator

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150, 150))


#fit model w data

history = model.fit(
            train_generator, # pass in the training generator
            steps_per_epoch=100,
            epochs=15,
            validation_data=validation_generator, # pass in the validation generator
            validation_steps=50,
            verbose=2
            )

model_json = model.to_json()
with open("./models/newmodel.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./models/newmodel.h5")
