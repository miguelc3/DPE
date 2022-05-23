import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATADIR = "..\..\..\imagens\\test_images\PointGrey\Dataset_pattern_3_4"
checkpoint_path = '..\\models\\model_5_best\\cp.ckpt'
IMG_SIZE = 299
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

# configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Picking a pre trained model
# -----------------------------
# Pre-trained model InceptionV3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet'
)

base_model.layers.pop()

# base_model.summary()
base_model.trainable = False

# Build the model
num_classes = len(class_names)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Dense(1024, activation='relu'),
  global_average_layer,
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Save best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=400,
  callbacks=[model_checkpoint_callback]
)

# Save the model
model.save('..\\models\\model_5')

# Unbatch dataset
val_ds = val_ds.unbatch()
test_images = list(val_ds.map(lambda x, y: x))
test_labels = list(val_ds.map(lambda x, y: y))

# Transform it into numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Evaluate the model
val_loss, val_acc = model.evaluate(test_images, test_labels, batch_size=1)
print('Validation accuracy = ' + str(val_acc))

# List all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



