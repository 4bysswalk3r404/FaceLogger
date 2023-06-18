import face_recognition
import timeit
import cv2
import dlib
from Lib.functions import *

image = cv2.imread('./group.png')
image = cv2.resize(image, (600, 400))
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

dlibdetector = dlib.get_frontal_face_detector()
cvdetector = cv2.CascadeClassifier("./Lib/haarcascade_frontalface_default.xml")

fr = lambda: face_recognition.face_locations(gray)
cv = lambda: cvdetector.detectMultiScale(image, minSize=(50, 50))
dl = lambda: dlibdetector.run(image, 1)

print("face recognition:    %f", timeit.timeit(fr, number=30))
print("opencv: %f" % timeit.timeit(cv, number=30))
print("dlib: %f" % timeit.timeit(dl, number=30))

frLocations = fr()
frimage = image.copy()
for frrect in frLocations:
    topleft, bottomright = faceCorners(frrect)
    cv2.rectangle(frimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('frimage', frimage)

cvLocations = cv()
cvimage = image.copy()
for cvrect in cvLocations:
    topleft, bottomright = cv2Corners(cvrect)
    cv2.rectangle(cvimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('cvimage', cvimage)

dlLocations, _, _ = dl()
dlimage = image.copy()
for dlrect in dlLocations:
    topleft, bottomright = dlibCorners(dlrect)
    cv2.rectangle(dlimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('dlimage', dlimage)

cv2.waitKey(0)