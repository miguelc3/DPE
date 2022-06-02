import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob

# Global Variables
class_names = ['AMOLGADELAS', 'OK', 'RISCOS']
num_classes = len(class_names)


def load_model(path_ckpt):

    IMG_SHAPE = (224, 224, 3)
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                                         include_top=False,
                                                         weights='imagenet'
                                                         )
    base_model.layers.pop()
    base_model.trainable = False

    # Build the model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(path_ckpt)

    return model


def single_predict(model, path_img, IMG_SIZE):

    # Show image to predict - just for visualization
    img = cv2.imread(path_img)
    # cv2.imshow('IMAGE TO TEST', img)
    # cv2.waitKey(0)

    img = image.load_img(path_img, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediction -> higher probability
    prediction = model.predict(img_preprocessed)
    class_predict = class_names[np.argmax(prediction[0])]
    print(prediction)

    return class_predict


def multi_predict(model, DATADIR, nr_pred):

    preds = []
    counter = 0
    for image_file in glob.iglob(DATADIR):
        counter += 1
        prediction = single_predict(model, image_file, 224)
        preds.append(prediction)
        print(prediction)

        if counter == nr_pred:
            break



def main():
    # Import tensorflow model
    # model = tf.keras.models.load_model('../../models/RISCOS_AMOLG/model_1')

    path_ckpt = '../models/2_RISCOS_AMOLG/model_2_best/cp.ckpt'
    model = load_model(path_ckpt)

    DATADIR = "../../../imagens/DATASETS_pg/testes/pattern_40/*.png"
    IMG_PATH = "../../../imagens/DATASETS_pg/testes/pattern_40/img1.png"  # Path for image to predict

    # prediction = single_predict(model, IMG_PATH, 224)
    # print(prediction)
    multi_predict(model, DATADIR, 24)


if __name__ == '__main__':
    main()
