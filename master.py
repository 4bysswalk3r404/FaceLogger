import os
import cv2
import face_recognition
import json
import numpy as np
import functions

class Master:
    def __init__(self):
        self.names = []
        self.encodings = []

    def _addEncoding(self, name, encoding):
        self.names.append(name)
        self.encodings.append(encoding)

    def addEncodingImage(self, img, name):
        encoding = face_recognition.face_encodings(img)[0]
        self.encodings.append(encoding)
        self.names.append(name)
    
    def unknownCount(self):
        count = 0
        for name in self.names:
            if 'unknown' in name.lower():
                count += 1
        return count

    def compareImg(self, img):
        try:
            encoding = face_recognition.face_encodings(img)
            encoding = encoding[0]
        except Exception as e:
            print(e)
            return str(e)

        face = face_recognition.compare_faces(self.encodings, encoding)
        if True not in face:
            name = f'unknown{self.unknownCount()}'
            self._addEncoding(name, encoding)
        else:
            index = np.argmax(face)
            name = self.names[index]
        return name


def encode_compare(frame, bottomRight, topLeft, known_encodings):
    bottomR = (bottomR[0], bottomR[1] + 40)
    topL = (topL[0], topL[1] - 20)

    face = functions.getImageCorners(frame, topL, bottomR)
    face = cv2.resize(face, (256, 256))
    cv2.imshow("face", face)
    try:
        encoding = face_recognition.face_encodings(face)[0]
        compare = face_recognition.compare_faces(known_encodings, encoding)
        print(compare)
    except Exception as e:
        print(e)

def main():
    import dlib
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    known_encodings = [face_recognition.face_encodings(cv2.imread('face.png'))[0]]
    known_names = ['connor']

    lastcenter = (0,0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rects, scores, idx = detector.run(gray, 1)
        for rect in rects:
            topL, bottomR = functions.rect_to_corners(rect)

            width = (bottomR[0] - topL[0]) // 2 + topL[0]
            height = (bottomR[0] - topL[0]) // 2 + topL[0]

            cv2.rectangle(frame, bottomR, topL, (255, 0, 0))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()
