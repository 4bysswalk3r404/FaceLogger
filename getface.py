import cv2
from Lib.functions import getImage

faceDetector = cv2.CascadeClassifier(".\\Lib\\haarcascade_frontalface_default.xml")

image = cv2.imread('presidents.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

rects = faceDetector.detectMultiScale(gray)

for i, rect in enumerate(rects):
    face = getImage(image, *rect)
    face = cv2.resize(face, (256, 256))
    cv2.imshow(str(i), face)
    cv2.imwrite(f"./{str(i)}.png", face)

cv2.waitKey(0)