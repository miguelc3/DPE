import numpy as np
import cv2
import glob
import os


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    # _, img = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(path_img, path_save):

    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    # proc_img = bin(equ(img))
    proc_img = bin(img)

    file_name = 'ok1_bin.png'
    cv2.imwrite(os.path.join(path_save, file_name), proc_img)

    cv2.imshow('IMG', proc_img)
    cv2.waitKey(0)


def main():

    path_img = "..\..\..\imagens\\test_images\PointGrey\\testes\\ok1.png"
    path_save = "..\..\..\imagens\\test_images\PointGrey\\testes"
    process(path_img, path_save)


if __name__ == "__main__":
    main()
