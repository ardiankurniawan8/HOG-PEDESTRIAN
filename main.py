from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in paths.list_images("images"):
    image = cv2.imread(imagePath)
    # cv2.imshow("image", image)
    image = imutils.resize(image,width=min(400, image.shape[1]))

    original = image.copy()
    # cv2.imshow("original", original)

    (rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(8,8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(original, (x,y),(x+w,y+h),(255,0,255),2)
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # print(rects)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA,yA),(xB,yB),(0,255,0),2)
    
    filename = imagePath[imagePath.rfind("/")+1:]
    print("[INFO]{}: {} original boxes, {} after suppression".format(filename, len(rects),len(pick)))

    cv2.imshow("BEFORE", original)
    cv2.imshow("AFTER", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()