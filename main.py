import cv2
import dlib
import face_recognition
from functions import *

from master import Master
import math

p = r"D:\Zone\prog\Neural\Image\facelogger\Lib\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# face_encoder = dlib.face_recognition_model_v1(r"D:\Zone\prog\Neural\Image\facelogger\Lib\dlib_face_recognition_resnet_model_v1.dat")

objects = {}

master = Master()
cap = cv2.VideoCapture(0)
count = 0

lastCenter = (0,0)
fps = FrameRate()
lastname = 'error'
name = 'unknown'
while True:
    _, frame = cap.read()
    showframe = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    rects, scores, idx = detector.run(gray, 1)
    scores = [round(score, 4) for score in scores]

    deltatime = fps.deltaTime()
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        bottomRight, topLeft = getRect(shape)
        width = bottomRight[0] - topLeft[0]
        height = bottomRight[1] - topLeft[1]
        x = (width) // 2 + topLeft[0]
        y = (height) // 2 + topLeft[1]
        center = (x,y)

        #============================

        cv2.circle(showframe, center, 3, (255,0,0))
        cv2.circle(showframe, lastCenter, 3, (0,150,150))

        cv2.line(showframe, bottomRight, topLeft, (150, 150, 0))
        cv2.line(showframe, center, lastCenter, (150, 0, 150))

        facesize = math.sqrt(width * width + height * height)
        centerDist = math.dist(center, lastCenter)
        cv2.putText(showframe, f"{centerDist:.2f}", center, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        # if math.dist(key, center) <= deltatime * (int(facesize) >> 1):

        #============================

        mult = 2 if len(rects) == 1 else 1
        if not centerDist <= mult * deltatime * facesize:
            try:
                bottomRight = (bottomRight[0] + 20, bottomRight[1] + 40)
                topLeft = (topLeft[0] - 20, topLeft[1] - 40)
                face = getImageCorners(frame, topLeft, bottomRight)
                face = cv2.resize(face, (150, 150))
                cv2.imshow('face', face)
                encoding = face_recognition.face_encodings(face)[0]
                result = face_recognition.compare_faces(master.encodings, encoding)
                index = np.argmax(result)
                name = master.names[index]
            except Exception as e:
                print(e)
        else:
            name = lastname
        lastCenter = center
        lastname = name

        cv2.rectangle(showframe, bottomRight, topLeft, (0,0,0), 2)
        cv2.putText(showframe, name, topLeft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    fps.showFPS(showframe)
    cv2.imshow('frame', showframe)

    if cv2.waitKey(1) & 0xFF == 27:
        break