import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
cv2.namedWindow("Foreground",1)
bgs = cv2.BackgroundSubtractorMOG()

while(1):
    img = cap.read()[1]
    if img is not None:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #img = cv2.equalizeHist(img)
        fgmask = bgs.apply(img)
        cv2.imshow("Foreground",fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
