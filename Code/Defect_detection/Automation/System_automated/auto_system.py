import threading
import cv2
import PySpin
import matplotlib.pyplot as plt
import sys
import keyboard
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
from MESConnection import MESConnection
import socket
import datetime


# Global variables
class_names_pattern = ['AMOLGADELAS', 'OK', 'RISCOS', 'PO']
class_names_no_pattern = ['FALTA_TINTA', 'OK']
global model_pattern
global model_no_pattern
global counter_images
global secs
global nioBit_pattern
global nioBit_no_pattern
global cycle_count
cycle_count = 0

global continue_recording
continue_recording = True


def handle_close(evt):

    global continue_recording
    continue_recording = False


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):

    global continue_recording

    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Close program
        print('Press enter to close the program..')

        # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
        fig = plt.figure(1)

        # Close the GUI when close event happens
        fig.canvas.mpl_connect('close_event', handle_close)

        # Retrieve and display images
        while continue_recording:
            try:

                #  Retrieve next received image
                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    # print('Image incomplete with image status %d ...' % image_result.GetImageStatus())  # change - commented
                    pass

                else:

                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()

                    global secs
                    t = time.time()
                    if t - secs > 8:
                        got_image(image_data)  # GET IMAGE TO ANALYSE EVERY 8 SECONDS
                        secs = time.time()

                    # Draws an image on the current figure
                    plt.imshow(image_data, cmap='gray')  # CHANGE - commented this line

                    # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # Interval is in seconds.
                    plt.pause(0.001)

                    # Clear current reference of a figure. This will improve display speed significantly
                    plt.clf()

                    # If user presses enter, close the program
                    if keyboard.is_pressed('ENTER'):
                        print('Program is closing...')

                        # Close figure
                        plt.close('all')
                        input('Done! Press Enter to exit...')
                        continue_recording = False

                        #  Release image
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):

    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


# =====================================================================================================
# Functions added by me
def load_models(checkpoint_pattern, checkpoint_no_pattern):

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
    model_pattern.load_weights(checkpoint_pattern)

    # MODEL WITHOUT PATTERN
    num_classes_no_pattern = len(class_names_no_pattern)

    model_no_pattern = tf.keras.Sequential([
        base_model,
        global_average_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes_no_pattern, activation='softmax')
    ])

    # The model weights (that are considered the best) are loaded into the model.
    model_no_pattern.load_weights(checkpoint_no_pattern)

    return model_pattern, model_no_pattern


def got_image(img):

    # Receive image
    print('Got image')

    image_pil = Image.fromarray(img)  # convert numpy.ndarray to PIL image

    plt.figure(2)
    plt.imshow(image_pil, cmap='gray')
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

    # Send image to be analysed for model with or without pattern
    global counter_images
    counter_images += 1
    if counter_images % 2 != 0:
        predict_pattern(image_pil)
    else:
        predict_no_pattern(image_pil)


def divide_image(img_original):
    # Divide original image in 4 images of 450 x 450
    img_tl = img_original[0:450, 0:450]  # Image top left
    img_tr = img_original[0:450, 358:]  # Image top right
    img_bl = img_original[158:, 0:450]  # Image bottom left
    img_br = img_original[158:, 358:]  # Image bottom right

    imgs = [img_tl, img_tr, img_bl, img_br]
    return imgs


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(pil_image):

    # convert pil image to opencv
    opencv_img = np.array(pil_image)

    # Process image
    proc_img = bin(equ(opencv_img))
    # proc_img = bin(img)

    return proc_img


def predict(model, pattern, img, IMG_SIZE):

    # Check  witch classes to predict in
    if pattern == 'y':
        class_names = class_names_pattern
    elif pattern == 'n':
        class_names = class_names_no_pattern

    # Show image to predict - just for visualization
    # plt.imshow(img, cmap='gray')
    # plt.show(block=False)
    # plt.pause(1.5)
    # plt.close()

    new_size = (IMG_SIZE, IMG_SIZE)
    img = img.resize(new_size)
    img = img.convert('RGB')

    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediction -> higher probability
    prediction = model.predict(img_preprocessed)
    class_predict = class_names[np.argmax(prediction[0])]

    if prediction[0][np.argmax(prediction[0])] < 0.9:
        class_predict = 'not_sure'

    class_cert = prediction[0][np.argmax(prediction[0])] * 100
    if class_predict == 'not_sure':
        print('Prediction: note sure because confidence is low (' + str(class_cert) + '). Assumed OK.')
    else:
        print('Prediction: ' + class_predict + ' with ' + str(class_cert) + '% confidence')

    return class_predict


def feedback(color, msg):

    # predefined value of bgr
    bgr = (255, 255, 255)

    if color == 'red':
        bgr = (0, 0, 255)
    elif color == 'green':
        bgr = (0, 255, 0)

    # Create a blank 300x300 black image
    image = np.zeros((800, 1200, 3), np.uint8)

    # Fill image with red color(set each pixel to red)
    image[:] = bgr

    cv2.imshow(msg, image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def predict_pattern(image_pil):
    global nioBit_pattern
    print('Predicting with pattern')

    # Preprocess Image
    proc_img = process(image_pil)
    frame = np.asarray(proc_img)  # Image to array to divide

    # Divide image in four sub images
    imgs = divide_image(frame)

    i = 0
    for img in imgs:
        i += 1

        img = Image.fromarray(img)  # Convert array to PIL image
        prediction = predict(model_pattern, 'y', img, 224)

        if prediction == 'RISCOS':  # prediction is "riscos" -> break cycle
            nioBit_pattern = 1

            msg = 'SURFACE IS DAMAGED: RISCOS'
            feedback('red', msg)
            break

        elif prediction == 'AMOLGADELAS':  # prediction is "amolgadelas" -> break cycle
            nioBit_pattern = 4

            msg = 'SURFACE IS DAMAGED: AMOLGADELAS'
            feedback('red', msg)
            break

        elif prediction == 'PO':  # prediction is "PO" -> break cycle
            nioBit_pattern = 3

            msg = 'SURFACE IS DAMAGED: EXCESSO DE PÓ'
            feedback('red', msg)
            break

        elif prediction == 'not_sure':  # certainty too low -> image is assumed to be ok
            print('Unsure about the class, it is assumed to be ok.')

        if i == 4:
            nioBit_pattern = 0
            msg = 'SURFACE PATTERN IS OK'
            feedback('green', msg)


def predict_no_pattern(original_image):
    global nioBit_no_pattern

    print('Predicting without pattern')

    # Prepare image to divide
    frame = np.asarray(original_image)

    # Divide image in four sub images
    imgs = divide_image(frame)

    i = 0
    for img in imgs:
        i += 1

        img = Image.fromarray(img)  # Convert to PIL image
        prediction = predict(model_no_pattern, 'n', img, 224)

        if prediction == 'FALTA_TINTA':  # prediction is "falta tinta" -> break cycle
            nioBit_no_pattern = 2

            msg = 'SURFACE IS DAMAGED: FALTA TINTA'
            feedback('red', msg)
            break

        elif prediction == 'not_sure':  # certainty too low -> image is assumed to be ok
            print('Unsure about the class, it is assumed to be ok.')

        if i == 4:
            nioBit_no_pattern = 0
            msg = 'SURFACE NO PATTERN IS OK'
            feedback('green', msg)

    final_decision()


def final_decision():
    global nioBit_pattern
    global nioBit_no_pattern
    global cycle_count

    # Update cycle count
    cycle_count += 1

    if nioBit_pattern == 0 and nioBit_no_pattern == 0:
        print('Surface is ok by both models')
        # Send info to MES
        send_info_mes(result=1, nioBit=0, cycleCount=cycle_count)

    else:
        print('Surface is not ok')
        # Send info to MES
        nioBit = nioBit_pattern + nioBit_no_pattern

        send_info_mes(result=2, nioBit=nioBit, cycleCount=cycle_count)


def send_info_mes(result, nioBit, cycleCount):

    # Define header
    header = MESConnection.Header("1")
    locationHeader = MESConnection.LocationHeader("9999", "999", "1", "1", "1", "1", "1", "Defect_detection", "PC")

    resultHeader = MESConnection.ResultHeader(result=str(result), typeNo="999999999", workingCode="1",
                                              nioBits=str(nioBit), workCycleCount=str(cycleCount))

    # Build identifier
    currentDate = datetime.datetime.now()
    year = str(currentDate.year)
    month = str(currentDate.month)
    day = str(currentDate.day)
    hour = str(currentDate.hour)
    min = str(currentDate.minute)
    sec = str(currentDate.second)
    identifier = '8370_' + year + '_' + month + '_' + day + '_' + hour + '_' + min + '_' + sec + '_999999999'

    mesConnection = MESConnection(header, locationHeader, resultHeader, identifier, resHeadEnabled=True)

    message = mesConnection.CreateTelegram()

    # Transform the message into an array of bytes
    telegramBytes = mesConnection.BuildTelegram(message)

    # Open socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    target_port = 55065
    target_host = "localhost"
    s.connect((target_host, target_port))

    s.send(telegramBytes)  # Send bytes

    # MES response
    # Remove the first four bytes (they are just the size of the message)
    data = s.recv(1024).decode(encoding='utf-8', errors='ignore')
    print(data)

    telegramResult = mesConnection.ResultTelegram().ProcessResponse(data)
    print(telegramResult)

    time.sleep(1)

    s.close()  # Close socket every time a message is sent -> MES bug


# =====================================================================================================
def main():

    # ==================================================
    # Part added by me
    global counter_images
    global secs
    counter_images = 0
    secs = time.time()

    # Models checkpoints
    checkpoint_filepath_pattern = '../../models/4_RISCOS_AMOLG_PO/model_1_best/cp.ckpt'
    checkpoint_filepath_no_pattern = '../../models/3_FALTA_TINTA/model_1_best/cp.ckpt'

    # Load models
    global model_pattern, model_no_pattern
    model_pattern, model_no_pattern = load_models(checkpoint_filepath_pattern, checkpoint_filepath_no_pattern)

    # ==================================================

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    # input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
