import cv2


def skip_frames(cap, count):
    for i in range(count):
        cap.grab()


cap = cv2.VideoCapture('l_05_persons_0_smoke_1_01.mp4')

while (cap.isOpened()):
    ret, frame1 = cap.read()
    skip_frames(cap, 5)
    ret, frame2 = cap.read()
    orig_now_img = frame2.copy()

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
    result = orig_now_img
    cv2.imshow('res', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
