import cv2

import detectors.detector as detector
import videostream.objects as objects
from config import Config
from videostream.stream import skip_frames, draw_smoke


def execut(net=None):
    import time
    cap = cv2.VideoCapture("l_05_persons_0_smoke_1_01.mp4")
    sum = time.time() + 1
    out = None
    config = Config("{\"conditions\":[\
         {\"type\":0,\"operation\":2,\"value\":0,\"event\":0,\"area\":[0,0,1000,1000]},\
        {\"type\":1,\"operation\":2,\"value\":0,\"event\":0,\"area\":[0,0,1000,1000]},\
        {\"type\":2,\"operation\":2,\"value\":0,\"event\":0,\"area\":[0,0,1000,1000]},\
        {\"type\":3,\"operation\":2,\"value\":0,\"event\":0,\"area\":[0,0,1000,1000]}\
        ]}")
    while (cap.isOpened()):
        ret, frame1 = cap.read()
        skip_frames(cap, 4)
        ret, frame2 = cap.read()
        orig_now_img = frame2.copy()

        diffimg = detector.det_diff(frame1, frame2)
        counturs = detector.get_counters(diffimg, 20)

        # for detect motion
        # result = (detector.det_motions(orig_now_img, counturs, config))

        # # for detect smoke
        result = orig_now_img
        if (len(objects.smoke) == 0):
            result = detector.det_smoke(frame2, counturs, config)
        result = draw_smoke(result)

        # if time.time() > sum:
        #     sum = sum + 1
        #     blob = cv2.dnn.blobFromImage(frame2, size=(544, 320))
        #     net.setInput(blob)
        #     out = net.forward()
        #     # out = out["detection_out"]
        #     print(out.shape)
        #
        #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     for detect in out.reshape(-1, 7):
        #         confidence = detect[2]
        #         x_min = int(detect[3] * frame2.shape[1])
        #         y_min = int(detect[4] * frame2.shape[0])
        #         x_max = int(detect[5] * frame2.shape[1])
        #         y_max = int(detect[6] * frame2.shape[0])
        #         if confidence > 0.05:
        #             cv2.rectangle(frame2, (x_min, y_min), (x_max, y_max), (0, 255, 0))
        # try:
        #     for detect in out.reshape(-1, 7):
        #         confidence = detect[2]
        #         x_min = int(detect[3] * frame2.shape[1])
        #         y_min = int(detect[4] * frame2.shape[0])
        #         x_max = int(detect[5] * frame2.shape[1])
        #         y_max = int(detect[6] * frame2.shape[0])
        #         if confidence > 0.05:
        #             cv2.rectangle(frame2, (x_min, y_min), (x_max, y_max), (0, 255, 0))
        # except Exception:
        #     pass
        cv2.imshow('frame', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return

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
    # net2 = cv2.dnn.readNet("C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32\\pedestrian-detection-adas-0002.xml",
    #                        "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\pedestrian-detection-adas-0002\\FP32\\pedestrian-detection-adas-0002.bin")
    # net2 = cv2.dnn.readNet(
    #      "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.xml",
    #      "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.bin")
    # net = cv2.dnn.readNet(
    #     "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.xml",
    #     "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-vehicle-bike-detection-crossroad-0078\\FP32\\person-vehicle-bike-detection-crossroad-0078.bin")
    # net = cv2.dnn.readNet(
    #     "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml",
    #     "C:\\Intel\\computer_vision_sdk_2018.4.420\\deployment_tools\\intel_models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.bin")
    execut()
