import face_recognition
import cv2
import numpy as np

class Master:
    def __init__(self) -> None:
        self.encodings : list[np.ndarray] = []
        self.names     : list[str]        = []

    def compareToDatabase(self, image : cv2.Mat) -> str:
        encoding = face_recognition.face_encodings(image)[0]
        face = face_recognition.compare_faces(self.encodings, encoding)
        index = np.argmax(face)
        if index >= len(self.names):
            name = 'unknown'
        else:
            name = self.names[index]
        return name, True

    def addEncoding(self, image : cv2.Mat, name : str) -> None:
        encoding = face_recognition.face_encodings(image)

        self.encodings.append(encoding)
        self.names.append(name)

        cv2.imwrite(f'./faces/{name}.png', image)

if __name__ == "__main__":
    master = Master()
    master.addEncoding()