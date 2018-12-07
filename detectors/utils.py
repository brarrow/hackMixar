import cv2
import numpy as np
from PIL import Image


def get_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def get_img(path):
    pil_image = Image.open(path).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def drawbb(img, x, y, w, h, danger):
    if danger == 0:
        color = (0, 215, 255)
    elif danger == 1:
        color = (0, 140, 255)
    else:
        color = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color)
    return img
