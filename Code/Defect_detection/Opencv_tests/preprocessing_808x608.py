import numpy as np
import cv2
import glob
import os


def devide_img(path_imgs):

    imgs = []
    for img_file in glob.iglob(path_imgs):

        # Load image
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        # Divide original image in 4 images of 450 x 450
        img_tl = img[0:450, 0:450]  # Image top left
        img_tr = img[0:450, 358:]  # Image top right
        img_bl = img[158:, 0:450]  # Image bottom left
        img_br = img[158:, 358:]  # Image bottom right

        imgs.extend((img_tl, img_tr, img_bl, img_br))


    return imgs


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    # _, img = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return img


def equ(image):
    img = cv2.equalizeHist(image)
    return img


def process(imgs, path_save):
    c = 1
    for img in imgs:
        # Equalize histogram
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        proc_img = bin(equ(img))
        # proc_img = equ(img)

        # Save processed image to file
        file_name = "img" + str(c) + ".png"
        cv2.imwrite(os.path.join(path_save, file_name), proc_img)
        c += 1


def main():
    # Path for images to process
    path_imgs = "..\..\..\imagens\\DATASETS_pg\\SpinView_trigger\\*.png"

    # Path for save images
    path_save = "..\..\..\imagens\\DATASETS_pg\\testes\\pattern_40"

    imgs = devide_img(path_imgs)
    process(imgs, path_save)


if __name__ == "__main__":
    main()
