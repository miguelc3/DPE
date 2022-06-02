import cv2
import os
from skimage import io


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    # _, img = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(img, path_save, save_name):
    proc_img = bin(equ(img))
    # proc_img = bin(img)
    cv2.imwrite(os.path.join(path_save, save_name), proc_img)
    return proc_img


def main():

    path_img = "..\..\..\imagens\\DATASETS_pg\\Falta_tinta\\6.png"
    path_save = "..\..\..\imagens\\DATASETS_pg\\Falta_tinta"

    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    process(img, path_save, '6_preproc.png')


if __name__ == "__main__":
    main()
