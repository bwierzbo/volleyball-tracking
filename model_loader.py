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
    return np.argmax(loaded_model.predict(img, verbose = 0))



#model = keras.models.load_model('../volleyball-tracking/model/model.h5')
json_file = open('../volleyball-tracking/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('../volleyball-tracking/model/model.h5')

