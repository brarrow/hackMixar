import cv2
import matplotlib.pylab as plt
import numpy as np
from PIL import Image


def normalizeLight(img):
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


def diff(img1, img2):
    img1_norml = normalizeLight(img1)
    img2_norml = normalizeLight(img2)

    img1b = blur(img1_norml, 10)
    img2b = blur(img2_norml, 10)
    diff = cv2.absdiff(img1b, img2b)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 20
    imask = mask > th

    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]

    cv2.imwrite("result.png", canvas)
    return canvas


def getimg(path):
    pil_image = Image.open(path).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def blur(img, size=5):
    kernel = np.ones((size, size), np.float32) / size ** 2
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def det_smoke(img):
    res = blur(img, 2)
    plt.imshow(res)
    plt.show()
    cv2.imwrite('smokedet.png', res)


def test_smoke():
    imgpath = "scene1.png"
    img = getimg(imgpath)
    img = normalizeLight(img)
    det_smoke(img)


def test_diff():
    imgpath1 = "scene1.png"
    imgpath2 = "scene2.png"

    img1 = getimg(imgpath1)
    img2 = getimg(imgpath2)

    diff(img1, img2)


test_diff()
