import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# DATADIR = "..\..\imagens\\test_images\PointGrey\\new_defect"
DATADIR = "..\..\imagens\\test_images\PointGrey\\Dataset_pattern_3_2"
IMG_SIZE = 224
batch_size = 1

# Data augmentation object
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=0.5,
        validation_split=0.2,
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

# -----------------------------
# Picking a pre trained model
# -----------------------------
# Pre-trained model MobileNet V2
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')

base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                                     include_top=False,
                                                     weights='imagenet'
                                                     )

base_model.summary()

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
num_classes = 2
class_names = ['nok', 'ok']

# Create the model
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
  train_generator,
  validation_data=validation_generator,
  epochs=50
)

# Save model
model.save('../models/model_4')



