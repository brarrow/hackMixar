import cv2

import detectors.detector as detector
# import detectors.people_hog_detector as pepdet
import detectors.utils as utils


def skip_frames(cap, count):
    for i in range(count):
        cap.grab()


cap = cv2.VideoCapture('vc_07_smoke_1_02.mp4')

hog = utils.get_hog()
while (cap.isOpened()):
    ret, frame1 = cap.read()
    skip_frames(cap, 4)
    ret, frame2 = cap.read()
    orig_now_img = frame2.copy()
    diffimg = detector.det_diff(frame1, frame2)
    counturs = detector.get_counters(diffimg, 20)

    # for detect motion
    # result = detector.det_motions(orig_now_img, counturs)

    # for detect smoke
    result = detector.det_smoke(orig_now_img, counturs)

    # human
    # frame2 = detector.det_diff(frame1, frame2)
    # result = frame2
    # counters = detector.get_counters(frame2, 10)
    # humans = []
    # for counter in counters:
    #     x, y, w, h = counter[:4]
    #     bufimg = frame2[y:y + h, x:x + w]
    #     plt.imshow(bufimg)
    #     plt.show()
    #     humans.append(pepdet.detectHuman(bufimg, hog))
    # result = orig_now_img
    # for hum in humans:
    #     result = pepdet.drawBox(result, hum)

    cv2.imshow('res', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
