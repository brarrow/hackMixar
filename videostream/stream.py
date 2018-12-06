import cv2

from detectors import detector


def skip_frames(cap, count):
    for i in range(count):
        cap.grab()


cap = cv2.VideoCapture('l_05_persons_0_smoke_1_01.mp4')

while (cap.isOpened()):
    ret, frame1 = cap.read()
    skip_frames(cap, 3)
    ret, frame2 = cap.read()

    result = detector.det_motions(frame1, frame2)

    cv2.imshow('res', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
