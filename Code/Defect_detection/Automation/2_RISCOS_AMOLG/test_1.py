import cv2
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import glob
from time import sleep


# Global variables
class_names = ['AMOLGADELAS', 'OK', 'RISCOS']

def predict(model, path_img, IMG_SIZE):

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
    if prediction[0][np.argmax(prediction[0])] < 0.9:
        class_predict = 'not_sure'
    else:
        print(class_predict)

    return class_predict


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(img, path_save, save_name):

    proc_img = bin(equ(img))
    # proc_img = bin(img)
    cv2.imwrite(os.path.join(path_save, save_name), proc_img)
    return proc_img


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def feedback(color, msg):

    if color == 'red':
        bgr = (0, 0, 255)
        win_name = msg
    elif color == 'green':
        bgr = (0, 255, 0)
        win_name = msg

    # Create a blank 300x300 black image
    image = np.zeros((800, 1200, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    image[:] = bgr

    cv2.imshow(win_name, image)
    cv2.waitKey(3000)


def main():

    # Directories
    DATADIR = "../../../../imagens/DATASETS_pg/SpinView_trigger\*.png"
    checkpoint_filepath = '../../models/2_RISCOS_AMOLG/model_2_best/cp.ckpt'
    path_save = "../images_saved"

    # Load best model - higher accuracy
    IMG_SIZE = 224
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
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
        global_average_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)

    for image_file in glob.iglob(DATADIR):
        print(image_file)
        sleep(2)

        img_original = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        # print('Original image shape = ' + str(img_original.shape))

        # Divide original image in 4 images of 450 x 450
        img_tl = img_original[0:450, 0:450]  # Image top left
        img_tr = img_original[0:450, 358:]  # Image top right
        img_bl = img_original[158:, 0:450]  # Image bottom left
        img_br = img_original[158:, 358:]  # Image bottom right

        imgs = [img_tl, img_tr, img_bl, img_br]

        try:
            latest_nr = newest(path_save)[13:-4]
            a = int(latest_nr) + 1
        except:
            a = 1

        print(a)

        i = 0
        for img in imgs:
            file_name = str(a) + '.jpg'
            proc_img = process(img, path_save, file_name)
            img_pred = newest(path_save)
            print(img_pred)
            prediction = predict(model, img_pred, IMG_SIZE)
            if prediction == 'RISCOS':
                msg = 'SURFACE IS DAMAGED: RISCOS'
                print(msg)
                feedback('red', msg)
                break
            elif prediction == 'AMOLGADELAS':
                msg = 'SURFACE IS DAMAGED: AMOLGADELAS'
                print(msg)
                feedback('red', msg)
                break

            elif prediction == 'not_sure':
                print('Unsure about the class, it is assumed to be ok.')

            # Update variables
            a += 1
            i += 1

            if i == 4:
                msg = 'SURFACE IS OK'
                print(msg)
                feedback('green', msg)


if __name__ == '__main__':
    main()
