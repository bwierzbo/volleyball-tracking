#!/usr/bin/env python

"""model_loader.py: Running this code will load the model trained from train.py for use by ballornotball.py."""

__author__      = "Benjamin Wierzbanowski"


import sys
import numpy as np
import cv2 as cv
import os

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

from keras.models import load_model
from keras.models import model_from_json
from keras.utils import img_to_array
from tensorflow import keras

size = 32
dim = 3


json_file = open('./models/newmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./models/newmodel.h5")




def checkIMG(pic):
    img = cv.resize(pic, (size, size))
    img = np.reshape(img,[1,size, size, dim])
    #prediction = np.argmax(loaded_model.predict(img, verbose=0), axis=-1)[0]
    prediction = int(loaded_model.predict(img, verbose=0)[0])
    return int(loaded_model.predict(img, verbose=0)[0])