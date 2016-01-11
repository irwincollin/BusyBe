import time
import datetime

import urllib

import itertools
import numpy as np
import cv2

# see github repo for project description


# function Takes a matrix of images. Each image is scaled to fit within a portion of a total combined image.
# expects RGB. To convert gray to RGB use colored_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
# authored by Collin Irwin
# TODO: bug or oddity where the width and height don't actually match screen resolution (i.e. 1400 width fills my 2800 screen)
# TODO: make images contiguous instead of having large gaps
# TODO: add splitting so that a 6x2 could be split into two 3x2 to take advanteg of aspect ratios etc
# TODO: lines between rows/columns
def combine_image_rows(rows, total_width, total_height, transpose=False):
    if transpose:
        rows = list(itertools.izip_longest(*rows))
    num_rows = len(rows)
    num_images = len(rows[0])
    height_per_image = total_height / float(num_rows)
    width_per_image = total_width / float(num_images)

    ret_image = np.zeros((total_height, total_width, 3), np.uint8)

    row_num = 0
    for row in rows:
        img_num = 0
        for row_img in row:
            if row_img is None:
                continue
            height, width, channels = row_img.shape
            height_scalar = height_per_image / float(height)
            width_scalar = width_per_image / float(width)
            scalar = height_scalar if height_scalar < width_scalar else width_scalar
            scalar -= 0.001 # TODO: this is a temporary hack to make sure it is below the size and note 1 pixel above
            row_img = cv2.resize(row_img, (0, 0), fx=scalar, fy=scalar)

            height, width, channels = row_img.shape
            height_start = (int)(height_per_image*row_num)
            width_start = (int)(width_per_image*img_num)
            ret_image[height_start : (height_start + height), width_start : (width_start + width), :3] = row_img

            img_num += 1
        row_num += 1

    return ret_image


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

# setup cascades
path = "cascades/haarcascades/"
faceCascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
profileCascade = cv2.CascadeClassifier(path + "haarcascade_profileface.xml")
fullBodyCascade = cv2.CascadeClassifier(path + "haarcascade_fullbody.xml")
upperBodyCascade = cv2.CascadeClassifier(path + "haarcascade_upperbody.xml")
lowerBodyCascade = cv2.CascadeClassifier(path + "haarcascade_lowerbody.xml")

# instantiate useful classes
detector = cv2.SimpleBlobDetector()

# Cams and can data
cams = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
cam_count = len(cams)
cam_range = range(cam_count)
failures = [0 for _ in cam_range]
refreshes = [0 for _ in cam_range]
t_minus = [None for _ in cam_range]
t = [None for _ in cam_range]
t_plus = [None for _ in cam_range]

# Process cams
while 1:
    cam_num = 0
    diff = [None for _ in cam_range]
    img = [None for _ in cam_range]
    for cam in cams:
        img_url = "http://recsports.ufl.edu/cam/" + cam + ".jpg"
        try:
            filename, header = urllib.urlretrieve(img_url)
        except urllib.ContentTooShortError:
            continue
        img[cam_num] = cv2.imread(filename)

        if img[cam_num] is None:
            print "Failed to load image from " + img_url
            break

        failures[cam_num] = 0
        refreshes[cam_num] += 1

        gray = cv2.cvtColor(img[cam_num], cv2.COLOR_BGR2GRAY)

        # saving images to detect diff over time
        t_minus[cam_num] = t[cam_num]
        t[cam_num] = t_plus[cam_num]
        t_plus[cam_num] = gray

        if t_minus[cam_num] is not None and t[cam_num] is not None and t_plus[cam_num] is not None:
            diff[cam_num] = diff_img(t_minus[cam_num], t[cam_num], t_plus[cam_num])
            ret, diff[cam_num] = cv2.threshold(diff[cam_num], 40, 255, cv2.THRESH_BINARY)
            # cv2.countNonZero can be used for counting number of white pixels potentially

        cam_num += 1

    # Display the resulting images
    image_rows = []
    for cam_num in range(len(cams)):
        if img[cam_num] is None:
            continue

        image_row = [img[cam_num]]
        if diff[cam_num] is not None:
            image_row.append(cv2.cvtColor(diff[cam_num], cv2.COLOR_GRAY2RGB))
        image_rows.append(image_row)

    final_image = combine_image_rows(image_rows, 1400, 350, True)
    cv2.imshow('All Cameras', final_image)

    # Record the diff amount for each cam
    now = datetime.datetime.now()
    nowStr = now.strftime("%Y-%m-%d %H:%M:%S")
    # to read: dt.strptime(nowStr, "%Y-%m-%d %H:%M:%S")
    for cam_num in range(len(cams)):
        diffRating = cv2.countNonZero(diff[cam_num])
        with open(cams[cam_num] + ".txt", "a") as camfile:
            camfile.write(nowStr + "\n")

    if cv2.waitKey(20000) & 0xFF == ord('q'):
        pass


# TODO: reincorporate failure handling and improve hnadling
#else:
#    print "Failure #" + str(failures)
#    failures += 1
#    time.sleep(5)
#    if failures > 120:
#        print "10 minutes of failures! Program exiting"
#        break
