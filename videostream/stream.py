import cv2

import detectors.detector as detector
# import detectors.people_hog_detector as pepdet
import detectors.utils as utils
import videostream.objects as objects


def skip_frames(cap, count):
    for i in range(count):
        cap.grab()


def draw_smoke(image):
    objects.smoke = detector.maximize_counters(objects.smoke)
    for el in objects.smoke:
        countur = el[:4]
        danger = el[4]
        image = utils.drawbb(image, *countur[:4], danger)
    return image


# with danger at the end

hog = utils.get_hog()

# x, y, w, h, danger(0/1/2)
counters_to_draw = []

def init_stream(config):
    cap = cv2.VideoCapture('videostream/l_05_persons_0_smoke_1_01.mp4')
    while (cap.isOpened()):
        ret, frame1 = cap.read()
        skip_frames(cap, 4)
        ret, frame2 = cap.read()
        orig_now_img = frame2.copy()
        diffimg = detector.det_diff(frame1, frame2)
        counturs = detector.get_counters(diffimg, 20)

        # for detect motion
        result = detector.det_motions(orig_now_img, counturs, config.get_cond("3"))

        # for detect smoke
        result = detector.det_smoke(orig_now_img, counturs, config.get_cond("2"))
        result = draw_smoke(result)

        cv2.imshow('res', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
