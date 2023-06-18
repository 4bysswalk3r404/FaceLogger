import cv2
from Lib.functions import getImage
import face_recognition
from numpy import argmax
import master as Master

faceDetector = cv2.CascadeClassifier(".\\Lib\\haarcascade_frontalface_default.xml")

image = cv2.imread('presidents.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

rects = faceDetector.detectMultiScale(gray)

master = Master.Master()
master.addEncoding(cv2.imread('.\\obama.png'), 'obama')

for i, rect in enumerate(rects):
    topleft = rect[2:]

    face = getImage(image, *rect)
    face = cv2.resize(face, (256, 256))

    name, contained = master.compareToDatabase(face)
    del face

    cv2.rectangle(image, rect, (0,0,0))
    cv2.putText(image, name, topleft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("image", image)
cv2.waitKey(0)