import cv2
import numpy as np


def dilatate(img, size=5, iterations=2):
    kernel = np.ones((size, size), np.uint8)
    dilatation = cv2.dilate(img, kernel, iterations)
    return dilatation


def normalize_light(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def erode(img, size=5, iterations=2):
    kernel = np.ones((size, size), np.uint8)
    erosion = cv2.erode(img, kernel, iterations)
    return erosion


def sharp(img, size):
    kernel = np.ones((size, size), np.uint8)
    dilatation = cv2.dilate(img, kernel, iterations=2)
    erosion = cv2.erode(dilatation, kernel, iterations=1)
    return erosion


def blur(img, size=5):
    kernel = np.ones((size, size), np.float32) / size ** 2
    dst = cv2.filter2D(img, -1, kernel)
    return dst
