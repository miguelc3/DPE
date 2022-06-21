import tensorflow as  tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob
from PIL import Image

# Global Variables
class_names = ['AMOLGADELAS', 'OK', 'RISCOS', 'PO']
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


def single_predict(model, img, IMG_SIZE):

    # Show image to predict - just for visualization
    # img.show()

    new_size = (IMG_SIZE, IMG_SIZE)
    img = img.resize(new_size)
    img = img.convert('RGB')

    img_batch = np.expand_dims(img, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediction -> higher probability
    prediction = model.predict([img_preprocessed])

    class_predict = class_names[np.argmax(prediction[0])]
    print(prediction)
    print(class_predict)

    class_cert = prediction[0][np.argmax(prediction[0])] * 100
    print(class_cert)
    print('Prediction: ' + class_predict + ' with ' + str(class_cert) + '% confidence')

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


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(path):

    # Load image in opencv
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    proc_img = bin(equ(img))
    return proc_img


def main():

    path_ckpt = '../models/4_RISCOS_AMOLG_PO/model_1_best/cp.ckpt'
    model = load_model(path_ckpt)

    # DATADIR = "../../../imagens/DATASETS_pg/testes/pattern_40/*.png"
    IMG_PATH = "../../../imagens/Defeitos/Riscos/riscos_40/1.png"  # Path for image to predict

    proc_img = process(IMG_PATH)
    img = Image.fromarray(proc_img)

    prediction = single_predict(model, img, 224)
    # print(prediction)
    # multi_predict(model, DATADIR, 24)


if __name__ == '__main__':
    main()
