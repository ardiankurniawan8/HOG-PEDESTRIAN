from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,255),2)

    cv2.imshow("result", frame)
    press = cv2.waitKey(25)
    if press == ord('q'):
        break;
cv2.destroyAllWindows()