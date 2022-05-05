import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


DATADIR = "..\..\imagens\\test_images\PointGrey\Dataset_pattern_3_2"
# path = "..\..\imagens\\test_images\PointGrey\save_ds"
IMG_SIZE = 450
batch_size = 1


class_names = ['nok', 'ok']

# Data augmentation object
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2
        )

train_generator = train_datagen.flow_from_directory(
        DATADIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True,
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        DATADIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True,
        subset='validation')

num_classes = 2

# Create the model
model = tf.keras.Sequential([
  # Preprocessing layers
  # resize_and_rescale,
  # data_augmentation,
  # Rest of the model
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


# Making predictions
model.save('../models/model_3')


