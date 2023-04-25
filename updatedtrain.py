from keras.models import Sequential, Model
from keras.layers import Conv2D, Convolution2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Rescaling
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
import PIL
import PIL.Image
import tensorflow as tf

FAST_RUN = False
IMAGE_WIDTH=32
IMAGE_HEIGHT=32
batch_size = 15

train_ds = tf.keras.utils.image_dataset_from_directory(
  "../volleyball-tracking/training/data/train/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "../volleyball-tracking/training/data/validation/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2
size = 32
input_shape = (size, size, 3)

# model = Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   Conv2D(32, 3, activation='relu', input_shape=input_shape),
#   MaxPooling2D(),
#   Conv2D(64, 3, activation='relu'),
#   MaxPooling2D(),
#   Flatten(),
#   Dense(64, activation='relu'),
#   Dropout(0.1),
#   Dense(num_classes, activation='softmax')
# ])

# model.compile(
#     optimizer= SGD(lr=0.01),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"])


model = Sequential([
  Rescaling(1./255, input_shape=(size, size, 3)),
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10
)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")