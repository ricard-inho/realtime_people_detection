import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/crowdhuman_yolov5m.pt')

detec = model("mercury.jpeg")

for x1,y1,x2,y2,conf,obj_type in detec.xyxy[0]:
    print(obj_type)

import numpy as np
import cv2

cap = cv2.VideoCapture('rtsp://admin:0202master1965@75.110.254.189:8554/streaming/channels/101.sdp')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

