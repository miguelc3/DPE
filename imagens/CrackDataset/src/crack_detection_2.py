#!/usr/bin/env python3

"""
Architecture: Pre-trained model (MobileNetV2);
              1 global average layer (GlobalAveragePooling2D);
              1 Dense layer (activation='softmax');

val_accuracy (15 epochs) = 0.8868
"""

import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

keras = tf.keras

DATADIR = "../DATASET/IMAGES"
IMG_SIZE = 160
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
print(first_image.shape)
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Picking a pre trained model
# -----------------------------
# Pre-trained model MobileNet V2
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

# Build the model
num_classes = 5
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
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










