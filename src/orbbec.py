from cgitb import enable
import cv2
from halcon import tuple_greater_equal_elem_s
from primesense import openni2
from primesense import _openni2 as c_api
import time
import numpy as np

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


if color_stream.camera is not None:
    color_stream.camera.set_auto_exposure(bAutoExposure)
    color_stream.camera.set_exposure(exposure)
    # color_stream.camera.set_auto_white_balance(bAutoExposure)

time.sleep(1)

img = color_stream.read_frame()

frame_data = img.get_buffer_as_uint8()
colorPix = np.frombuffer(frame_data, dtype=np.uint8)
colorPix.shape = (height, width, 3)

print(type(colorPix), " ", len(colorPix), " ", colorPix[0])

colorPix = cv2.cvtColor(colorPix, cv2.COLOR_RGB2BGR)
cv2.imshow("win", colorPix)
cv2.waitKey(0)
