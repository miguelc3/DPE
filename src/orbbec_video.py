import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import os
import time

frameRate = 30  # frames per second
width = 640  # Width of image
height = 480

bAutoExposure = True
exposure = 400

# redistPath = "../Codigo_VS2/OpenNI-Windows-x64-2.3.0.55/Redist/"
redistPath = "../Redist/"

openni2.initialize(redistPath)  # The OpenNI2 Redist folder

# Open a device
dev = openni2.Device.open_any()
# Check to make sure it's not None

color_stream = dev.create_color_stream()
color_stream.start()
color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                               resolutionX=width, resolutionY=height, fps=frameRate))

while True:
    frame = color_stream.read_frame()
    frame_data = frame.get_buffer_as_uint8()
    img = np.frombuffer(frame_data, dtype=np.uint8)
    img.shape = (height, width, 3)
    # img = np.concatenate((img, img, img), axis=0)
    # img = np.swapaxes(img, 0, 2)
    # img = np.swapaxes(img, 0, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", img)

    path = 'C:\\Users\\User\\Desktop\\Uni\\DPE\\test_images'
    # path = 'C:\\Users\\User\\Desktop\\Uni'

    t = time.time()
    img_name = 'img_' + str(t) + '.jpg'
    print(img_name)

    cv2.imwrite(os.path.join(path, img_name), img)

    cv2.waitKey(0)

openni2.unload()
