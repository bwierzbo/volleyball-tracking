# Disable warning
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd 
import tensorflow

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
import os
print(os.listdir("../VolleyballTracking/"))

filenames = os.listdir("../VolleyballTracking/coloroutpath/")
categories = []

for filename in filenames:
    category = filename.split('-')[0]
    if category == 'notball':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(df.head())