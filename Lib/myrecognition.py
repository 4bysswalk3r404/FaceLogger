import face_recognition
import cv2
import numpy as np

def getRect(rect):
    return (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])

class Face:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename

class Master:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def addEncoding(self, face: Face):
        myface = face_recognition.load_image_file(face.filename)
        myfaceencoding = face_recognition.face_encodings(myface)[0]

        self.known_face_encodings.append(myfaceencoding)
        self.known_face_names.append(face.name)
    def addEncoding(self, name : str, filename : str):
        self.addEncoding(Face(name, filename))

    def compareAll(self, img : cv2.Mat) -> tuple :
        """gives the name in index one, and the face location in index 2"""
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        names = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_location = tuple([z * 4 for z in face_location])
            name = self.compare(face_encoding)
            names.append((name, face_location))
        return names

    def compare(self, face_encoding):
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
        return name


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)

    master = Master()

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        matches = master.compareAll(small_frame)

        for match in matches:
            location = getRect(match[1])
            print(match, location)
            cv2.rectangle(frame, location[0], location[1], (0,0,0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()