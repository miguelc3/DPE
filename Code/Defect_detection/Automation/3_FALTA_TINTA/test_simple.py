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
    # print(class_predict)

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
    DATADIR_pattern = "../../../../imagens/DATASETS_pg/SpinView_trigger/pattern/3_ok.png"
    DATADIR_no_pattern = "../../../../imagens/DATASETS_pg/SpinView_trigger/no_pattern/1_ok.png"

    # Path to models weights
    checkpoint_filepath_pattern = '../../models/2_RISCOS_AMOLG/model_2_best/cp.ckpt'
    checkpoint_filepath_no_pattern = '../../models/3_FALTA_TINTA/model_1_best/cp.ckpt'

    # Path to save divided images
    path_save_pattern = "saved_imgs_pattern"
    path_save_no_pattern = "saved_imgs_no_pattern"

    # Load best model - higher accuracy
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
    model_no_pattern.load_weights(checkpoint_filepath_no_pattern)

    # ======================================
    # MAKE PREDICTIONS WITH PATTERN MODEL
    # ======================================

    # Load image
    img_original = cv2.imread(DATADIR_pattern, cv2.IMREAD_GRAYSCALE)

    # Divide original image in 4 images of 450 x 450
    imgs = divide_image(img_original)

    # Get the latest img nr in directory
    try:
        latest_nr = newest(path_save_pattern)[19:-4]
        print(latest_nr)
    except:
        a = 1

    # Initialize variables
    i = 0
    pattern_ok = False

    # Analyse each image at a time
    for img in imgs:
        file_name = str(a) + '.jpg'

        # Preprocess image
        process(img, path_save_pattern, file_name)
        img_pred = newest(path_save_pattern)
        print(img_pred)

        # Make predictions
        prediction = predict(model_pattern, 'y', img_pred, IMG_SIZE)
        if prediction == 'RISCOS':  # prediction is "riscos" -> break cycle
            msg = 'SURFACE IS DAMAGED: RISCOS'
            print(msg)
            feedback('red', msg)
            break

        elif prediction == 'AMOLGADELAS':  # prediction is "amolgadelas" -> break cycle
            msg = 'SURFACE IS DAMAGED: AMOLGADELAS'
            print(msg)
            feedback('red', msg)
            break

        elif prediction == 'OK':  # Image is predicted to be ok
            print('OK')

        elif prediction == 'not_sure':  # certainty too low -> image is assumed to be ok
            print('Unsure about the class, it is assumed to be ok.')

        # Update variables
        a += 1
        i += 1

        if i == 4:
            pattern_ok = True
            print('SURFACE PATTERN IS OK')

    # ======================================
    # MAKE PREDICTIONS WITH NO PATTERN MODEL
    # ======================================

    # Load image
    img_original = cv2.imread(DATADIR_no_pattern, cv2.IMREAD_GRAYSCALE)

    # Divide original image in 4 images of 450 x 450
    imgs = divide_image(img_original)

    try:
        latest_nr = newest(path_save_no_pattern)[22:-4]
        a = int(latest_nr) + 1
    except:
        a = 1

    # Initialize variables
    i = 0
    no_pattern_ok = False
    for img in imgs:
        file_name = str(a) + '.jpg'

        # Save divided image
        cv2.imwrite(os.path.join(path_save_no_pattern, file_name), img)

        # Get latest image saved in directory
        img_pred = newest(path_save_no_pattern)
        print(img_pred)

        # Make prediction -> function predict
        prediction = predict(model_no_pattern, 'n', img_pred, IMG_SIZE)

        if prediction == 'FALTA_TINTA':  # prediction is "falta tinta" -> break cycle
            msg = 'SURFACE IS DAMAGED: FALTA TINTA'
            print(msg)
            feedback('red', msg)
            break

        elif prediction == 'OK':  # Image is predicted to be ok
            print('OK')

        elif prediction == 'not_sure':  # certainty too low -> image is assumed to be ok
            print('Unsure about the class, it is assumed to be ok.')

        # Update variables
        a += 1
        i += 1

        # If cycle don't break -> image is ok
        if i == 4:
            no_pattern_ok = True
            msg = 'SURFACE NO PATTERN IS OK'
            print(msg)

    if pattern_ok and no_pattern_ok:
        msg = 'Surface is ok from both models'
        print(msg)
        feedback('green', msg)
    else:
        print('Surface is not ok')


if __name__ == '__main__':
    main()
