import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Loading dataset
DATADIR = "..\..\..\imagens\\test_images\PointGrey\Dataset_pattern_3_4"
IMG_SIZE = 450
batch_size = 1

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

# Unbatch dataset
val_ds = val_ds.unbatch()
test_images = list(val_ds.map(lambda x, y: x))
test_labels = list(val_ds.map(lambda x, y: y))

# Transform it into numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Loading model
model = tf.keras.models.load_model('../models/model_1')
val_loss, val_acc = model.evaluate(test_images,  test_labels, batch_size=1)

print(val_acc)


