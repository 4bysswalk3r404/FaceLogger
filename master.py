import face_recognition
import cv2
import numpy as np
import os.path
from os import walk

class Master:
    def __init__(self) -> None:
        self.encodings : list[np.ndarray] = []
        self.names     : list[str]        = []

    def initDatabase(self, facesDirectory : str) -> bool:
        filepath = next(os.walk(facesDirectory))
        for file in filepath[2]:
            path = os.path.join(facesDirectory, file)
            face = cv2.imread(path)
            face = cv2.resize(face, (256, 256))
            name = os.path.splitext(file)[0]
            print(f'adding {name} from {path} to database...')
            self._addEncoding(face, name)

    def getNumUnknown(self):
        for i, name in enumerate(self.names):
            if not f'unknown{i}' in name.lower():
                return f'unknown{i}'

    def compareToDatabase(self, image : cv2.Mat) -> str:
        encoding = face_recognition.face_encodings(image)
        try:
            encoding = encoding[0]
        except:
            pass
        name = self.getNumUnknown()
        try:
            face = face_recognition.compare_faces(self.encodings, encoding)
        except Exception as e: 
            print(e)
            return name, False
        index = np.argmax(face)
        if index < len(self.names):
            name = self.names[index]
        return name, True

    def _addEncoding(self, image : cv2.Mat, name : str) -> None:
        encoding = face_recognition.face_encodings(image)

        self.encodings.append(encoding)
        self.names.append(name)

    def addEncoding(self, image : cv2.Mat, name : str) -> None:
        self._addEncoding(image, name)
        cv2.imwrite(f'./faces/{name}.png', image)

if __name__ == "__main__":
    master = Master()
    master.addEncoding()