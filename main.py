import urllib
import time
import cv2
import numpy as np


def diff_img(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


def get_parts(image, cascade):
    parts = cascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return parts


def draw_parts(parts, color, onImage):
    for (x, y, w, h) in parts:
        cv2.rectangle(onImage, (x, y), (x + w, y + h), color, 2)


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
detector = cv2.SimpleBlobDetector()

# TODO: handle multiple cam feeds and place them in panes
# Cam number and image path
cam = "cam2"
emptyCam = cv2.imread("control - uf rec/" + cam + "[3].jpg")

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
        i += 1

        # saving images to detect diff over time
        t_minus = t
        t = t_plus
        t_plus = gray

        diff = None
        if t_minus is not None and t is not None and t_plus is not None:
            diff = diff_img(t_minus, t, t_plus)
            ret, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            keypoints = detector.detect(diff)
            diff = cv2.drawKeypoints(diff, keypoints, np.array([]), (255, 255, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # TODO: find body part moving average over past 20 checks
        # Find body parts
        faces = get_parts(img, faceCascade)
        profiles = get_parts(img, profileCascade)
        fullBodies = get_parts(img, fullBodyCascade)
        lowerBodies = get_parts(img, lowerBodyCascade)
        upperBodies = get_parts(img, upperBodyCascade)

        # Draw rectangles around the body parts
        draw_parts(fullBodies, (255, 128, 128), img)
        draw_parts(upperBodies, (255, 255, 0), img)
        draw_parts(lowerBodies, (255, 0, 255), img)
        draw_parts(faces, (0, 255, 0), img)
        draw_parts(profiles, (0, 0, 255), img)

        print "Parts Found: " + str(len(upperBodies)) + ", " + str(len(lowerBodies)) + ", " + str(
            len(faces)) + ", " + str(len(profiles))

        # TODO: record some measure of activity based on the diff
        # Display the resulting images
        if diff is not None:
            cv2.imshow('diff', diff)
        cv2.imshow('faces', img)

        if cv2.waitKey(20000) & 0xFF == ord('q'):
            break
    else:
        print "Failure #" + str(failures)
        failures += 1
        time.sleep(5)
        if failures > 120:
            print "10 minutes of failures! Program exiting"
            break
