import cv2

cap = cv2.VideoCapture("traffic.mp4")
cap.set(3,640)
cap.set(4,480)
classNames= []     
classFile= "/home/sahil/Desktop/Project/OpenCV/project/labels.txt"

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img=cap.read()

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    for classIds, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[classIds-1].upper(),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        cv2.destroyAllWindows()
        break
