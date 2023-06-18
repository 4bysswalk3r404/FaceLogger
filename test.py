import cv2
import dlib
import Lib.functions as functions
import master as Master

image = cv2.imread('./group.png')
image = functions.scaleImageMaxTo(image, 1000)
showimage = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

master = Master.Master()
master.initDatabase('./faces')

p = ".\\Lib\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)
dlibdetector = dlib.get_frontal_face_detector()
print('finishing initializing detectors')

dlLocations, scores, _ = dlibdetector.run(image, 1)
scores = [round(score, 4) for score in scores]
for i, rect in enumerate(dlLocations):
    topleft, bottomright = functions.dlibCorners(rect)
    cv2.rectangle(showimage, topleft, bottomright, (0,0,0), 2)

    shape = predictor(gray, rect)
    landmarks = functions.landmarksToPoints(shape)
    for landmark in landmarks:
        cv2.circle(showimage, landmark, 1, (256, 0, 0))
    cv2.putText(showimage, str(scores[i]), topleft, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
 
    face = functions.getImage(image, topleft, bottomright)
    face = cv2.resize(face, (256, 256))
    cv2.imshow(f'face{i}', face)
    name, _ = master.compareToDatabase(face)
    cv2.putText(showimage, name, (topleft[0], bottomright[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
 
    # cv2.imwrite(f'./faces/face{i}.png', face)

cv2.imshow('showimage', showimage)

cv2.waitKey(0)