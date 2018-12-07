import cv2



def execut(net):
    import time
    cap = cv2.VideoCapture("detectors/l_07_persons_0_02.mp4")
    sum = time.time() + 1
    out = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        # frame = frame[int(frame.shape[0]/2):frame.shape[0], int(frame.shape[1]/20):int(frame.shape[1]-1200)]
        # frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)), interpolation=cv2.INTER_AREA)
        # frame = frame[:int(frame.shape[0]/2)][:int(frame.shape[0]/2)]

        print(frame.shape)
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
                if confidence > 0.05:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
        try:
            for detect in out.reshape(-1, 7):
                confidence = detect[2]
                x_min = int(detect[3] * frame.shape[1])
                y_min = int(detect[4] * frame.shape[0])
                x_max = int(detect[5] * frame.shape[1])
                y_max = int(detect[6] * frame.shape[0])
                if confidence > 0.05:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
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
