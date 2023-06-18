import cv2
import dlib
import Lib.functions as functions
import master as Master

image = cv2.imread('./group.png')
showimage = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

p = ".\\Lib\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)
dlibdetector = dlib.get_frontal_face_detector()
print('finishing initializing detectors')

dlLocations, _, _ = dlibdetector.run(image, 1)
for rect in dlLocations:
    topleft, bottomright = functions.dlibCorners(rect)
    cv2.rectangle(showimage, topleft, bottomright, (0,0,0), 2)

    shape = predictor(gray, rect)
    landmarks = functions.landmarksToPoints(shape)
    for landmark in landmarks:
        cv2.circle(showimage, landmark, 3, (256, 0, 0))

showimage = functions.scaleImageMaxTo(showimage, 1000)
cv2.imshow('showimage', showimage)

cv2.waitKey(0)