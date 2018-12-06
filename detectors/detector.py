import cv2
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


def diff(img1, img2, th=20):
    img1_norml = normalizeLight(img1)
    img2_norml = normalizeLight(img2)
    img1b = blur(img1_norml, 10)
    img2b = blur(img2_norml, 10)

    img1_eros = erode(img1b, 10, 5)
    img2_eros = erode(img2b, 10, 5)

    diff = cv2.absdiff(img1_eros, img2_eros)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

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


def dilatate(img, size=5, iterations=2):
    kernel = np.ones((size, size), np.uint8)
    dilatation = cv2.dilate(img, kernel, iterations)
    return dilatation


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


def det_smoke(img1, img2):
    diffimg = diff(img1, img2)
    counturs = get_counters(diffimg)
    for countur in counturs:
        img2 = drawbb(img2, *countur)
    return img2


def findCountur(img, mx, mx_area, listConters):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        listConters.append([x, y, w, h])
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    return mx, mx_area, listConters


def drawbb(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 4)
    return img


def get_counters(img):
    mx = (0, 0, 0, 0)
    mx_area = 0
    listConters = []
    mx, mx_area, listConters = findCountur(img, mx, mx_area, listConters)
    return listConters

def test_smoke():
    imgpath1 = "nosteam.png"
    imgpath2 = "yessteam.png"

    img1 = getimg(imgpath1)
    img2 = getimg(imgpath2)

    res = det_smoke(img1, img2)
    cv2.imwrite('bbxs.jpg', res)
    


def test_diff():
    imgpath1 = "nosteam.png"
    imgpath2 = "yessteam.png"

    img1 = getimg(imgpath1)
    img2 = getimg(imgpath2)

    diff(img1, img2)


test_smoke()
