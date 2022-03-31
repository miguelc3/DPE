#!/usr/bin/env python3

"""
Architecture: No pre-trained model;
              3 conv layers (activation='relu');
              1 Dense layer (activation='softmax');

val_accuracy (15 epochs) = 0.9623
"""

import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

DATADIR = "../DATASET/IMAGES"
IMG_SIZE = 150
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
      DATADIR,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(IMG_SIZE, IMG_SIZE),
      batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
      DATADIR,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(IMG_SIZE, IMG_SIZE),
      batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# Normalize pixel values to be between 0 and 1
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the model
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  # add softmax layer
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Train the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=15
)







# training_data = []
#
# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)  # path to cats or dogs dir
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 training_data.append([new_array, class_num])
#             except Exception as e:
#                 pass
#
# create_training_data()
# random.shuffle(training_data)
# print(len(training_data))


