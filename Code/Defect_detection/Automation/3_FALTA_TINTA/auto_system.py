import cv2
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import glob
from time import sleep


# Global variables
class_names_pattern = ['AMOLGADELAS', 'OK', 'RISCOS']
class_names_no_pattern = ['FALTA_TINTA', 'OK']


def predict(model, pattern, path_img, IMG_SIZE):
    # Check  witch classes to predict in
    if pattern == 'y':
        class_names = class_names_pattern
    elif pattern == 'n':
        class_names = class_names_no_pattern

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


def divide_image(img_original):
    # Divide original image in 4 images of 450 x 450
    img_tl = img_original[0:450, 0:450]  # Image top left
    img_tr = img_original[0:450, 358:]  # Image top right
    img_bl = img_original[158:, 0:450]  # Image bottom left
    img_br = img_original[158:, 358:]  # Image bottom right

    imgs = [img_tl, img_tr, img_bl, img_br]
    return imgs


def main():

    # Path to raw images
    DATADIR_pattern = "../../../../imagens/DATASETS_pg/SpinView_trigger/pattern"
    DATADIR_no_pattern = "../../../../imagens/DATASETS_pg/SpinView_trigger/no_pattern"

    # Check initial nr of files in directories
    # Pattern
    try:
        nr_pics_pattern_old = len(os.listdir(DATADIR_pattern))
    except:
        nr_pics_pattern_old = 0

    # No pattern
    try:
        nr_pics_no_pattern_old = len(os.listdir(DATADIR_no_pattern))
    except:
        nr_pics_no_pattern_old = 0

    # Path to models weights
    checkpoint_filepath_pattern = '../../models/2_RISCOS_AMOLG/model_2_best/cp.ckpt'
    checkpoint_filepath_no_pattern = '../../models/3_FALTA_TINTA/model_1_best/cp.ckpt'

    # Path to save divided images
    path_save_pattern = "saved_imgs_pattern"
    path_save_no_pattern = "saved_imgs_no_pattern"

    # ============================================================================================
    # BUILD MODELS AND LOAD WEIGHTS
    # ====================================
    IMG_SIZE = 224
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                                         include_top=False,
                                                         weights='imagenet'
                                                         )
    base_model.layers.pop()
    base_model.trainable = False

    # Global average layer - same for both models
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # MODEL WITH PATTERN
    num_classes_pattern = len(class_names_pattern)

    model_pattern = tf.keras.Sequential([
        base_model,
        global_average_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes_pattern, activation='softmax')
    ])

    # The model weights (that are considered the best) are loaded into the model.
    model_pattern.load_weights(checkpoint_filepath_pattern)

    # MODEL WITHOUT PATTERN
    num_classes_no_pattern = len(class_names_no_pattern)

    model_no_pattern = tf.keras.Sequential([
        base_model,
        global_average_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes_no_pattern, activation='softmax')
    ])

    # The model weights (that are considered the best) are loaded into the model.
    model_pattern.load_weights(checkpoint_filepath_no_pattern)

    # ==============================================================================================

    # Main cycle
    while 1:
        # ================================
        # Run image though pattern model
        # ================================

        # Verify if there are new pictures in directory
        try:
            nr_pics_pattern = len(os.listdir(DATADIR_pattern))
        except:
            nr_pics_pattern = 0

        if nr_pics_pattern == nr_pics_pattern_old:
            print('There\'s no new picture yet')

        else:  # There is a new picture
            nr_pics_pattern_old = nr_pics_pattern
            latest_file_pattern = newest(DATADIR_pattern)

            # Load image
            path_img_pattern = latest_file_pattern

            # Divide image
            img_original_pattern = cv2.imread(path_img_pattern, cv2.IMREAD_GRAYSCALE)
            imgs_pattern = divide_image(img_original_pattern)

            # Get the nr of the last picture saved
            try:
                latest_nr = newest(path_save_pattern)[19:-4]
                a = int(latest_nr) + 1
            except:
                a = 1

            i = 0
            pattern_ok = False
            for img in imgs_pattern:
                file_name = str(a) + '.jpg'

                # Preprocess image
                process(img, path_save_pattern, file_name)
                img_pred = newest(path_save_pattern)
                print(img_pred)

                # Make prediction
                prediction_pattern = predict(model_pattern, 'y', img_pred, IMG_SIZE)

                if prediction_pattern == 'RISCOS':  # prediction is "riscos" -> break cycle
                    msg = 'SURFACE IS DAMAGED: RISCOS'
                    print(msg)
                    feedback('red', msg)
                    break

                elif prediction_pattern == 'AMOLGADELAS':  # prediction is "amolgadelas" -> break cycle
                    msg = 'SURFACE IS DAMAGED: AMOLGADELAS'
                    print(msg)
                    feedback('red', msg)
                    break

                elif prediction_pattern == 'OK':  # Image is predicted to be ok
                    print('OK')

                elif prediction_pattern == 'not_sure':
                    print('Unsure about the class, it is assumed to be ok.')

                # Update variables
                a += 1
                i += 1

                if i == 4:
                    pattern_ok = True
                    msg = 'SURFACE IS OK - PATTERN MODEL'
                    print(msg)
                    feedback('green', msg)

        # ================================
        # Run image though  no pattern model
        # ================================

        # Verify if there are new pictures in directory
        try:
            nr_pics_no_pattern = len(os.listdir(DATADIR_no_pattern))
        except:
            nr_pics_no_pattern = 0

        if nr_pics_no_pattern == nr_pics_no_pattern_old:
            print('There\'s no new picture yet')

        else:  # There is a new picture
            nr_pics_no_pattern_old = nr_pics_no_pattern
            latest_file_no_pattern = newest(DATADIR_no_pattern)

            # Load image
            path_img_no_pattern = latest_file_no_pattern

            # Divide image
            img_original_no_pattern = cv2.imread(path_img_no_pattern, cv2.IMREAD_GRAYSCALE)
            imgs_no_pattern = divide_image(img_original_no_pattern)

            # Get the nr of the last picture saved
            try:
                latest_nr = newest(path_save_no_pattern)[19:-4]
                a = int(latest_nr) + 1
            except:
                a = 1

            i = 0
            no_pattern_ok = False
            for img in imgs_no_pattern:
                file_name = str(a) + '.jpg'

                # Preprocess image
                process(img, path_save_no_pattern, file_name)
                img_pred = newest(path_save_no_pattern)
                print(img_pred)

                # Make prediction
                prediction_no_pattern = predict(model_no_pattern, 'y', img_pred, IMG_SIZE)

                if prediction_no_pattern == 'FALTA_TINTA':  # prediction is "falta tinta" -> break cycle
                    msg = 'SURFACE IS DAMAGED: FALTA TINTA'
                    print(msg)
                    feedback('red', msg)
                    break

                elif prediction_no_pattern == 'OK':  # Image is predicted to be ok
                    print('OK')

                elif prediction_no_pattern == 'not_sure':
                    print('Unsure about the class, it is assumed to be ok.')

                # Update variables
                a += 1
                i += 1

                if i == 4:
                    no_pattern_ok = True
                    msg = 'SURFACE IS OK - NO PATTERN MODEL'
                    print(msg)
                    feedback('green', msg)

        if pattern_ok and no_pattern_ok:  # surface is ok by both models
            print('SURFACE IS OK - BOTH MODELS')
        else:  # if one model predicts that surface is not ok
            print('SURFACE IS NOT OK')


if __name__ == '__main__':
    main()
