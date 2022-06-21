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


def process(path_imgs, path_save):
    c = 1
    for image_file in glob.iglob(path_imgs):
        # Equalize histogram
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        proc_img = bin(equ(img))
        # proc_img = equ(img)

        # Save processed image to file
        file_name = "img" + str(c) + ".png"
        cv2.imwrite(os.path.join(path_save, file_name), proc_img)
        c += 1


def main():
    # Path for images to process
    path_imgs = "..\..\..\imagens\\DATASETS_pg\\Excesso_po\\examples\\*.png"

    # Path for save images
    path_save = "..\..\..\imagens\\DATASETS_pg\\4_RISCOS_AMOLG_PO\\Dataset_40\\EXCESSO_PO"

    process(path_imgs, path_save)


if __name__ == "__main__":
    main()
