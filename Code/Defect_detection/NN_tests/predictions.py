import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2

# Import tensorflow model
model = tf.keras.models.load_model('../models/model_2_2')

IMG_SIZE = 224
class_names = ['nok', 'ok']

# Making predictions
IMG_PATH = "../../../imagens/test_images/PointGrey/testes/nok_3_2.png"  # Path for image to predict

# Show image to predict - just for visualization
img = cv2.imread(IMG_PATH)
cv2.imshow('IMAGE TO TEST', img)
cv2.waitKey(0)

img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

# plt.imshow(img_preprocessed)
# plt.show()

# Prediction -> higher probability
prediction = model.predict(img_preprocessed)
print(prediction)
print(class_names[np.argmax(prediction[0])])
