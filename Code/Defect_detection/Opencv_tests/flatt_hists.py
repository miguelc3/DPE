import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from PIL import Image
import os

path_imgs = "..\..\..\imagens\\test_images\PointGrey\\testes\*.bmp"
path_save = "..\..\..\imagens\\test_images\PointGrey\\a"

c = 1
for image_file in glob.iglob(path_imgs):
    # Equalize histogram
    img = cv.imread(image_file, 0)
    equ = cv.equalizeHist(img)

    # Just for visualization
    # hori = np.concatenate((img, equ), axis=1)
    # cv.imshow('IMG', hori)
    # cv.waitKey(0)

    # Save processed image to file
    file_name = "img" + str(c) + ".png"
    cv.imwrite(os.path.join(path_save, file_name), equ)
    c += 1
