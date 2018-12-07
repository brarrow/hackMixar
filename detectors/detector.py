import cv2
import matplotlib.pyplot as pyplot
import numpy as np

import detectors.people_hog_detector as pepdet
import detectors.utils as utils
import videostream.objects as objects
from detectors.utils import get_hog, drawbb
from filters.filters import dilatate, normalize_light, erode, blur

hog = get_hog()


def det_diff(img1, img2, th=20):
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

    return canvas


def det_motions(frame, counturs, config):
    conditions = config.conditions
    for i, cond in enumerate(conditions):
        if cond["type"] == 3:
            cond_counter = cond["area"]
            for j, countur in enumerate(counturs):
                if check_intercept(*cond_counter, *countur[:4]):
                    conditions[i]["isTrue"] = True
                    x, y, w, h = countur[:4]
                    frame = drawbb(frame, *countur[:4], cond["event"])
    return frame


def det_smoke(frame, counturs, config):
    conditions = config.conditions
    for i, cond in enumerate(conditions):
        if cond["type"] == 2:
            cond_counter = cond["area"]
            for j, countur in enumerate(counturs):
                # if check_intercept(*cond_counter, *countur[:4]):
                x, y, w, h = countur[:4]
                bboximg = frame[x:x + w, y:y + h]

                hist, tr1, tr2 = pyplot.hist(bboximg.mean(axis=2).flatten(), 255)
                res = np.mean(hist)
                if res > 75:
                    objects.smoke.append([*countur[:4], cond["event"]])
                    frame = drawbb(frame, *countur[:4], cond["event"])
    return frame


def det_human(img, danger):
    rects = pepdet.detectHuman(img, hog)
    res = utils.drawbb(img, rects, danger)
    return res


def counter_check_inside(counters):
    res = counters.copy()
    bres = [True] * len(res)
    length1 = len(counters)
    if length1 == 0:
        return counters
    for i in range(length1):
        counter1 = counters[i]
        x1, y1, w1, h1 = counter1[:4]
        j = i + 1
        while j < len(counters):
            counter2 = res[j]
            x2, y2, w2, h2 = counter2[:4]
            if (x2 > x1 and y2 > y1 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2):
                bres[j] = False
            j += 1
    res = []
    for i in range(length1):
        try:
            if bres[i]:
                res.append(counters[i])
        except TypeError:
            res.append(counters[i])
    return res


def check_intercept(x1, y1, w1, h1, x2, y2, w2, h2):
    intercept = False
    # rightup2
    rightupx2 = x2 + w2
    rightupy2 = y2

    # leftup2
    leftupx2 = x2
    leftupy2 = y2

    # rightdown2
    rightdownx2 = x2 + w2
    rightdowny2 = y2 + h2

    # leftdown2
    leftdownx2 = x2
    leftdowny2 = y2 + h2

    # rightup1
    rightupx1 = x1 + w1
    rightupy1 = y1

    # leftup1
    leftupx1 = x1
    leftupy1 = y1

    # rightdown1
    rightdownx1 = x1 + w1
    rightdowny1 = y1 + h1

    # leftdown1
    leftdownx1 = x1
    leftdowny1 = y1 + h1

    # for x
    if (rightupx2 > leftupx1 and rightupx2 < rightupx1):
        intercept = True
    if (leftupx2 > leftupx1 and leftupx2 < rightupx1):
        intercept = True
    if (rightdownx2 > leftupx1 and rightdownx2 < rightupx1):
        intercept = True
    if (leftdownx2 > leftupx1 and leftupx2 < rightupx1):
        intercept = True

    # for y
    if (rightupy2 < leftdowny1 and rightupy2 > rightupy1):
        intercept = True
    if (leftupy2 < leftdowny1 and leftupy2 > rightupy1):
        intercept = True
    if (rightdowny2 < leftdowny1 and rightdowny2 > rightupy1):
        intercept = True
    if (leftdowny2 < leftdowny1 and leftdowny2 > rightupy1):
        intercept = True
    return intercept


def maximize_counters(counters):
    for i in range(len(counters)):
        counter1 = counters[i]
        x1, y1, w1, h1 = counter1[:4]
        for j in range(len(counters)):
            counter2 = counters[j]
            intercept = False
            x2, y2, w2, h2 = counter2[:4]
            intercept = check_intercept(x1, y1, w1, h1, x2, y2, w2, h2)
            xn, yn, wn, hn = x2, y2, w2, h2
            if (intercept):
                xn, yn, wn, hn = min(x1, x2), min(y1, y2), max(w1, w2), max(h1, h2)
            counter2 = [xn, yn, wn, hn, wn * hn]
            counters[j] = counter2
    return counters


def find_counters(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    listConters = []

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > 6000:
            listConters.append([x, y, w, h, area])
    listConters = np.array(listConters)
    if (len(listConters) < 2):
        return listConters
    else:
        sortedConters = (listConters[np.argsort(listConters[:, 4])])[::-1]
        sortedConters = counter_check_inside(sortedConters)
        sortedConters = maximize_counters(sortedConters)
        return sortedConters


def get_nonblack(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    npimg = np.array(img)
    return npimg.any(axis=-1).sum()


def get_counters(img, count=5):
    listConters = find_counters(img)
    res = []
    if len(listConters) < count:
        return listConters
    else:
        for i in range(count):
            res.append(listConters[i])
        return res