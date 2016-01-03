import urllib
import time
import cv2
import numpy as np

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def getParts(image, cascade):
    parts = cascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return parts

def drawParts(parts, color, onImage):
    for (x, y, w, h) in parts:
        cv2.rectangle(onImage, (x, y), (x+w, y+h), color, 2)

# Create windows
cv2.namedWindow("diff", 0)
cv2.namedWindow("faces", 0)

# setup cascades
path = "cascades/haarcascades/"
faceCascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
profileCascade = cv2.CascadeClassifier(path + "haarcascade_profileface.xml")
fullBodyCascade = cv2.CascadeClassifier(path + "haarcascade_fullbody.xml")
upperBodyCascade = cv2.CascadeClassifier(path + "haarcascade_upperbody.xml")
lowerBodyCascade = cv2.CascadeClassifier(path + "haarcascade_lowerbody.xml")

# instantiate useful classes
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

        diff = None
        if t_minus is not None and t is not None and t_plus is not None:
            diff = diffImg(t_minus, t, t_plus)

# TODO: find body part moving average over past 20 checks
        if diff is not None:
            #Find body parts
            faces = getParts(diff, faceCascade)
            profiles = getParts(diff, profileCascade)
            fullBodies = getParts(diff, fullBodyCascade)
            lowerBodies = getParts(diff, lowerBodyCascade)
            upperBodies = getParts(diff, upperBodyCascade)
            
            # Draw rectangles around the body parts
            drawParts(fullBodies, (255, 128, 128), diff)
            drawParts(upperBodies, (255, 255, 0), diff)
            drawParts(lowerBodies, (255, 0, 255), diff)
            drawParts(faces, (0, 255, 0), diff)
            drawParts(profiles, (0, 0, 255), diff)

            print "Parts Found: " + str(len(upperBodies)) + ", " + str(len(lowerBodies))  + ", " + str(len(faces))  + ", " + str(len(profiles))

        # Display the resulting images
        if diff is not None:
            cv2.imshow('diff', diff)
        cv2.imshow('faces', img)
        
        if cv2.waitKey(20000) & 0xFF == ord('q'):
            break
    else:
        print "Failure #" + str(failures)
        failures = failures + 1
        time.sleep(5)
        if failures > 120:
            print "10 minutes of failures! Program exiting"
            break
