import cv2
import dlib
import face_recognition
from functions import *

import math

p = r"D:\Zone\prog\Neural\Image\facelogger\Lib\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
face_encoder = dlib.face_recognition_model_v1(r"D:\Zone\prog\Neural\Image\facelogger\Lib\dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)

best_score = 0

fps = FrameRate()
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    rects, scores, idx = detector.run(gray, 1)
    scores = [round(score, 4) for score in scores]

    showImg = frame.copy()

    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        bottomRight, topLeft = getRect(shape)
        width = bottomRight[0] - topLeft[0]
        height = bottomRight[1] - topLeft[1]
        x = (width) // 2 + topLeft[0]
        y = (height) // 2 + topLeft[1]
        center = (x,y)

        face = getImageRect(frame, rect)
        # cv2.putText(frame, f"{facesize:.2f}", (bottomRight[0]-10, topLeft[y]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(frame, f'face', topLeft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(showImg, bottomRight, topLeft, (0,0,0), 2)
        if scores[i] > best_score:
            color = (0, 255, 0) 
            best_score = scores[i]
            cv2.imwrite('face2.png', face)
            cv2.imshow('face2', face)
        else: color = (255, 0, 0)
        cv2.putText(showImg, f'{scores[i]:.2f}', topLeft, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
        cv2.circle(showImg, center, 3, (255,0,0))


    fps.showFPS(showImg)
    cv2.imshow('frame', showImg)

    if cv2.waitKey(1) & 0xFF == 27:
        break