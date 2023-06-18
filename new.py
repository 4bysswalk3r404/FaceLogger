import cv2
import face_recognition
from Lib.functions import *
from dataclasses import dataclass
from math import dist
import master as Master

@dataclass
class TrackingFaces:
    name : str
    center : tuple[2] | list[2]

p = ".\\Lib\\shape_predictor_68_face_landmarks.dat"
frontal_face_cascade = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier("./Lib/haarcascade_frontalface_default.xml")

predictor = dlib.shape_predictor(p)

master = Master.Master()
master.initDatabase('./faces')

cap = cv2.VideoCapture(".\\video1.mp4")
fps = FrameRate()
current_faces = []
lastCenters = []
while True:
    _, frame = cap.read()
    showFrame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    rects, scores, idx = frontal_face_cascade.run(gray, 1)
    scores = [round(score, 4) for score in scores]
    
    lastCenters_temp = []
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = rectFromLandmarks(shape)

        topLeft, bottomRight = dlibCorners(rect)

        #getting the center of the face
        width = bottomRight[0] - topLeft[0]
        height = bottomRight[1] - topLeft[1]
        x = (width) // 2 + topLeft[0]
        y = (height) // 2 + topLeft[1]
        center = (x,y)

        cv2.circle(showFrame, center, 3, (255,0,0))

        found = False
        for lastCenter in lastCenters:
            if dist(lastCenter, center) < 20:
                name = "known"
                found = True
                continue
        if not found:
            face = getImage(frame, rect)
            face = cv2.resize(face, (256, 256))
            name, masterKnown = master.compareToDatabase(face)
            if not masterKnown:
                master.addEncoding(face, name)

        cv2.rectangle(showFrame, topLeft, bottomRight, (0,0,0), 2)
        cv2.putText(showFrame, name, topLeft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        lastCenters_temp.append(center)
    lastCenters = lastCenters_temp

    fps.showFPS(showFrame)
    cv2.imshow('frame', showFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break