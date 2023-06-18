import cv2
import face_recognition
from Lib.functions import *
import master as Master


dlibdetector = dlib.get_frontal_face_detector()
cvdetector = cv2.CascadeClassifier("./Lib/haarcascade_frontalface_default.xml")

fr = lambda img: face_recognition.face_locations(img)
cv = lambda img: cvdetector.detectMultiScale(img, minSize=(50, 50))
dl = lambda img: (dlibdetector.run(img, 1))[0]

def loopFaces(showimage : cv2.Mat, 
            image : cv2.Mat,
            lambdaF : callable,
            translate : callable,
            color : tuple[3]
    ):
    rects = lambdaF(image)
    for rect in rects:
        topleft, bottomright = translate(rect)
        cv2.rectangle(showimage, topleft, bottomright, color, 2)

cap = cv2.VideoCapture(".\\video1.mp4")
fps = FrameRate()
while True:
    _, frame = cap.read()
    showFrame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    loopFaces(showFrame, gray, fr, faceCorners, (255, 0, 0))
    loopFaces(showFrame, frame, cv, cv2Corners, (0, 255, 0))
    loopFaces(showFrame, frame, dl, dlibCorners, (0, 0, 255))

    fps.showFPS(showFrame)
    cv2.imshow('frame', showFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

#Blue : face_recognition
#Green : cv2
#Red : dlib