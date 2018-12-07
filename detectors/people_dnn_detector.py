import cv2
import numpy as np
import matplotlib.pyplot as pyplot
from PIL import Image

import config
import videostream.objects as objects



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



def skip_frames(cap, count):
    for i in range(count):
        cap.grab()


def draw_smoke(image):
    objects.smoke = maximize_counters(objects.smoke)
    for el in objects.smoke:
        countur = el[:4]
        danger = el[4]
        image = drawbb(image, *countur[:4], danger)
    return image


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


def det_motions(frame, counturs, conditions):
    for cond in conditions:
        cond_counter = cond[:4]
        for countur in counturs:
            if check_intercept(*cond_counter, *countur[:4]):
                frame = drawbb(frame, *countur[:4], cond["event"])
    return frame


def det_smoke(frame, counturs, danger):
    for countur in counturs:
        x, y, w, h = countur[:4]
        bboximg = frame[x:x + w, y:y + h]
        hist, tr1, tr2 = pyplot.hist(bboximg.mean(axis=2).flatten(), 255)
        res = np.mean(hist)
        if res > 80:
            if danger > 0:
                objects.smoke.append([*countur[:4], danger])
            frame = drawbb(frame, *countur[:4], danger)
    return frame



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
            check_intercept(x1, y1, w1, h1, x2, y2, w2, h2)
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


def execut(net):
    import time
    cap = cv2.VideoCapture("detectors\l_05_persons_0_03.mp4")
    sum = time.time() + 1
    out = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        # frame = frame[int(frame.shape[0]/2):frame.shape[0], int(frame.shape[1]/20):int(frame.shape[1]-1200)]
        # frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)), interpolation=cv2.INTER_AREA)
        # frame = frame[:int(frame.shape[0]/2)][:int(frame.shape[0]/2)]
        ret, frame1 = cap.read()
        # skip_frames(cap, 4)
        ret, frame = cap.read()
        orig_now_img = frame.copy()
        diffimg = det_diff(frame1, frame)
        counturs = get_counters(diffimg, 20)

        # for detect motion
        result = det_motions(orig_now_img, counturs, config.get_cond("3"))

        # for detect smoke
        result = det_smoke(orig_now_img, counturs, config.get_cond("2"))
        result = draw_smoke(result)

        if time.time() > sum:
            sum = sum + 1
            blob = cv2.dnn.blobFromImage(frame, size=(544, 320))
            net.setInput(blob)
            out = net.forward()
            # out = out["detection_out"]
            print(out.shape)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for detect in out.reshape(-1, 7):
                confidence = detect[2]
                x_min = int(detect[3] * frame.shape[1])
                y_min = int(detect[4] * frame.shape[0])
                x_max = int(detect[5] * frame.shape[1])
                y_max = int(detect[6] * frame.shape[0])
                if confidence > 0.5:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
                cv2.rectangle(frame, (500, 400), (1300, 1100), (0, 0, 255), 5)
        try:
            for detect in out.reshape(-1, 7):
                confidence = detect[2]
                x_min = int(detect[3] * frame.shape[1])
                y_min = int(detect[4] * frame.shape[0])
                x_max = int(detect[5] * frame.shape[1])
                y_max = int(detect[6] * frame.shape[0])
                if confidence > 0.5:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
                cv2.rectangle(frame, (500, 400), (1300, 1100), (0, 0, 255), 5)
        except Exception:
            pass
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()
    # blob = cv2.dnn.blobFromImage(frame, size=(544, 320))
    # net.setInput(blob)
    # out = net.forward()
    # # out = out["detection_out"]
    # print(out.shape)
    #
    # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # for detect in out.reshape(-1, 7):
    #     confidence = detect[2]
    #     x_min = int(detect[3] * frame.shape[1])
    #     y_min = int(detect[4] * frame.shape[0])
    #     x_max = int(detect[5] * frame.shape[1])
    #     y_max = int(detect[6] * frame.shape[0])
    #     if confidence > 0.001:
    #         cv2.rectangle(frame, (x_min , y_min), (x_max, y_max),(0,255,0))
    #
    #     # cv2.imshow('frame', drawBox(gray, detectHuman(gray, hog)))
    #     cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # cap.release()

# cv2.waitKey()
if __name__ == '__main__':
    print("START")
    # from openvino.inference_engine import IENetwork, IEPlugin

    # Load a plugin.
    # plugin = IEPlugin('CPU')
    # frame = cv2.imread("2.PNG")

    # Load extensions library.
    # plugin.add_cpu_extension('cpu_extension_avx2.dll')
    # net = IENetwork.from_ir("C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml",
    #                          "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.bin")
    # net.reshape({'data': [1, 3, frame.shape[0], frame.shape[1]]})
    # Load IR to the plugin.
    # exec_net = plugin.load(net)
    # blob = cv2.dnn.blobFromImage(frame)# , size=(544, 320))
    # out = exec_net.infer(inputs={'data': blob})
    # help(net)

    frame = cv2.imread("4.PNG")
    net2 = cv2.dnn.readNet("C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32\\pedestrian-detection-adas-0002.xml",
                           "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32\\pedestrian-detection-adas-0002.bin")
    net2 = cv2.dnn.readNet(
         "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.xml",
         "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.bin")
    net = cv2.dnn.readNet(
        "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.xml",
        "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.bin")
    net = cv2.dnn.readNet(
        "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml",
        "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.bin")
    execut(net)
