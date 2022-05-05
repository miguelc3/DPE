import cv2
import numpy as np


def bin(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    # _, img = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return img


def main():
    path = "../../../imagens/test_images/PointGrey/testes/nok1.png"
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    image = bin(img_gray)
    filename = '../binary.jpg'
    cv2.imwrite(filename, image)

    # concatenate image Horizontally
    # hori = np.concatenate((img_gray, image), axis=1)
    cv2.imshow('Img', image)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
