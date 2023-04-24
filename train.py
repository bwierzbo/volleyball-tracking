# Disable warning
import warnings
warnings.filterwarnings('ignore')


import os
import random
import numpy as np
import pandas as pd 
import tensorflow

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import LeakyReLU
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

from keras.callbacks import EarlyStopping, ReduceLROnPlateau


size = 32


#Model creation

def createModel(input_shape, cls_n ):
    model = Sequential()

    activation = "relu"

    model = Sequential([
        Conv2D(32,(3,3), input_shape=input_shape),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(),
        Conv2D(64,(3,3)),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(),
        Flatten(),
        Dense(64),
        LeakyReLU(alpha=0.3),
        Dropout(0.1),
        Dense(2, activation='softmax')
      ])

    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


input_shape = (size, size, 3)

EPOCHS = 50
cls_n = 2


model = createModel(input_shape, cls_n)

train_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("train", color_mode="rgb", target_size = (size, size), batch_size = 32, class_mode = 'categorical')
model.fit_generator(training_set, steps_per_epoch = 20, epochs = EPOCHS, validation_steps = 10)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")