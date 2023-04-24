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



def checkIMG(img):
    img = cv.resize(img, (size, size))
    img = np.reshape(img,[1,size, size, dim])
    prediction = np.argmax(model.predict(img, verbose=0), axis=-1)[0]
    return prediction



model = keras.models.load_model('../volleyball-tracking/model/')

