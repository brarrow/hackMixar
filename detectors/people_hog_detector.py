import cv2
import numpy as np

def drawBox(frame, rects):
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in rects:
        # Tighten the rectangle around each person by a small margin
        shrinkW, shrinkH = int(0.05 * xB), int(0.15 * yB)
        cv2.rectangle(frame, (xA + shrinkW, yA + shrinkH), (xB - shrinkW, yB - shrinkH), 2)
    return frame

def detectHuman(image, hog):
    # (rects, weights) = hog.detectMultiScale(image, winStride=(5, 5), padding=(16, 16), scale=1.05, useMeanshiftGrouping=False)
    (rects, weights) = hog.detectMultiScale(image, winStride=(50, 50), padding=(16, 16), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    # pick = non_max_suppression(rects, probs=None) ?
    return rects

if __name__ == '__main__':
    print("START")
    # cap = cv2.VideoCapture(0)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # while (True):
        # Capture frame-by-frame
        # ret, frame = cap.read()
    frame = cv2.imread("1.jpg")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', drawBox(frame, detectHuman(frame, hog)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    # cap.release()

cv2.waitKey()
