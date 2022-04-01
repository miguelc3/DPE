#!/usr/bin/env python3

import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cifar10
import utils

TRAIN_DIR = "../DATASET/TRAIN_DIR"
TEST_DIR = "../DATASET/TEST_DIR"

IMG_SIZE = 160
batch_size = 15

class_names = []
num_classes = len(class_names)

# creates a data generator object that transforms images
datagen = ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = []
validation_generator = []

for img in train_ds:
    train_generator.append(datagen.flow(img, save_prefix='test', save_format='jpeg'))

for img in val_ds:
    validation_generator.append(datagen.flow(img, save_prefix='test', save_format='jpeg'))

print(train_generator.shape)



model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
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
  train_generator,
  validation_data=validation_generator,
  epochs=50
)
