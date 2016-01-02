import urllib
import time
import cv2
import numpy as np

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

# Create windows
cv2.namedWindow("diff", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("foreground", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("faces", cv2.CV_WINDOW_AUTOSIZE)

# setup cascades
path = "cascades/haarcascades/"
faceCascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
profileCascade = cv2.CascadeClassifier(path + "haarcascade_profileface.xml")
fullBodyCascade = cv2.CascadeClassifier(path + "haarcascade_fullbody.xml")
upperBodyCascade = cv2.CascadeClassifier(path + "haarcascade_upperbody.xml")
lowerBodyCascade = cv2.CascadeClassifier(path + "haarcascade_lowerbody.xml")

# instantiate useful classes
detector = cv2.SimpleBlobDetector()
bgs = cv2.BackgroundSubtractorMOG()

#Load and prep bgs with 0-person control image
cam = "cam2"
emptyCam = cv2.imread("control - uf rec/" + cam + "[3].jpg")
bgs.apply(emptyCam) # prep the bgs with empty image

failures = 0
i = 0
t_minus = None
t = None
t_plus = None

while i < 60:
    print "Downloading image number " + str(i);
    filename, header = urllib.urlretrieve("http://recsports.ufl.edu/cam/" + cam + ".jpg")
    img = cv2.imread(filename)
    
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        failures = 0
        i = i + 1

        # saving images to detect diff over time
        t_minus = t
        t = t_plus
        t_plus = gray

                   
        #Find Foreground - Potentially usable in counting number of people!
        fgmask = bgs.apply(img, 0.05)

# TODO: make body parts sections into functions
# TODO: find body part moving average over past 20 checks
        #Find body parts
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        profiles = profileCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        lowerBodies = lowerBodyCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        upperBodies = upperBodyCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        # Draw rectangles around the body parts
        for (x, y, w, h) in upperBodies:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        for (x, y, w, h) in lowerBodies:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in profiles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        print "Parts Found: " + str(len(upperBodies)) + ", " + str(len(lowerBodies))  + ", " + str(len(faces))  + ", " + str(len(profiles)) 

        # Display the resulting images
        
        if t_minus is not None and t is not None and t_plus is not None:
            #diff image with blob detection.
#TODO: fix blob detection...
            diff = diffImg(t_minus, t, t_plus)
            keypoints = detector.detect(diff)
            diff = cv2.drawKeypoints(diff, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("diff", diff)
        cv2.imshow('faces', img)
        cv2.imshow("foreground",fgmask)
        
        if cv2.waitKey(20000) & 0xFF == ord('q'):
            break
    else:
        print "Failure #" + str(failures)
        failures = failures + 1
        time.sleep(5)
        if failures > 120:
            print "10 minutes of failures! Program exiting"
            break
