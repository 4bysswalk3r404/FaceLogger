import cv2
import face_recognition
import numpy as np

known_face_encodings = [

]
known_face_names = [

]

face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
best_match_index = np.argmin(face_distances)
if matches[best_match_index]:
    name = known_face_names[best_match_index]
