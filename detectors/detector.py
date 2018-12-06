import cv2
import numpy as np
from PIL import Image

from filters.filters import dilatate, normalize_light, erode, blur


def diff(img1, img2, th=20):
    img1buf = img1.copy()
    img2buf = img2.copy()

    img1buf = normalize_light(img1buf)
    img2buf = normalize_light(img2buf)
    img1buf = blur(img1buf, 10)
    img2buf = blur(img2buf, 10)

    img1buf = erode(img1buf, 10, 5)
    img2buf = erode(img2buf, 10, 5)

    diff = cv2.absdiff(img1buf, img2buf)
    diff = dilatate(diff, 5, 3)
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


def det_motions(img1, img2):
    diffimg = diff(img1, img2)
    counturs = get_counters(diffimg, 5)
    for countur in counturs:
        img2 = drawbb(img2, *countur[:4])
    return img2


def find_counters(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    listConters = []

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > 40:
            listConters.append([x, y, w, h, area])
    listConters = np.array(listConters)
    if (len(listConters) < 2):
        return listConters
    else:
        sortedConters = (listConters[np.argsort(listConters[:, 4])])[::-1]
        return sortedConters


def drawbb(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 4)
    return img


def get_counters(img, count=5):
    listConters = find_counters(img)
    res = []
    if len(listConters) < count:
        return listConters
    else:
        for i in range(count):
            res.append(listConters[i])
        return res


def test_smoke():
    imgpath1 = "noman.png"
    imgpath2 = "yesman.png"

    img1 = getimg(imgpath1)
    img2 = getimg(imgpath2)

    res = det_motions(img1, img2)
    cv2.imwrite('bbxs.jpg', res)
    

def test_diff():
    imgpath1 = "nosteam.png"
    imgpath2 = "yessteam.png"

    img1 = getimg(imgpath1)
    img2 = getimg(imgpath2)

    diff(img1, img2)
