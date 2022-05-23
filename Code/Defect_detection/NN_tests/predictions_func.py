import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2


def main():
    # Import tensorflow model
    model = tf.keras.models.load_model('../models/model_4')
    IMG_PATH = "../../../imagens/test_images/PointGrey/testes/ok_3_2.png"  # Path for image to predict

    prediction = predict(model, IMG_PATH)
    print(prediction)


def predict(model, path_img):
    IMG_SIZE = 299
    class_names = ['nok', 'ok']

    # Show image to predict - just for visualization
    img = cv2.imread(path_img)
    cv2.imshow('IMAGE TO TEST', img)
    cv2.waitKey(0)

    img = image.load_img(path_img, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediction -> higher probability
    prediction = model.predict(img_preprocessed)
    class_predict = class_names[np.argmax(prediction[0])]
    print(prediction)

    return class_predict


if __name__ == '__main__':
    main()
