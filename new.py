import cv2
import face_recognition
from Lib.functions import *

p = ".\\Lib\\shape_predictor_68_face_landmarks.dat"
frontal_face_cascade = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier("./Lib/haarcascade_frontalface_default.xml")

predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(".\\video1.mp4")
fps = FrameRate()
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    rects, scores, idx = frontal_face_cascade.run(gray, 1)
    scores = [round(score, 4) for score in scores]
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        topLeft, bottomRight = rect_to_corners(rect)

        cv2.rectangle(frame, topLeft, bottomRight, (0,0,0), 2)
        cv2.putText(frame, "str(scores[i])", topLeft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    fps.showFPS(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break