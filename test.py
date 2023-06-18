import face_recognition
import timeit
import cv2
import dlib

image = cv2.imread('./group.png')
image = cv2.resize(image, (600, 400))
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

dlibdetector = dlib.get_frontal_face_detector()
cvdetector = cv2.CascadeClassifier("./Lib/haarcascade_frontalface_default.xml")

fr = lambda: face_recognition.face_locations(gray)
cv = lambda: cvdetector.detectMultiScale(rgb, minSize=(50, 50))
dl = lambda: dlibdetector.run(rgb, 1)

def dlibCorners(rect) -> tuple[2]:
    """takes a rect returned from dlib and returns 
    the top left corner, and the bottom right corner"""
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (left, top), (right, bottom)

def faceCorners(rect) -> tuple[2]:
    topleft, bottomright = (rect[3], rect[0]), (rect[2], rect[1])
    return topleft, bottomright

def cv2Corners(rect) -> tuple[2]:
    """cv2's detectMultiScale return a list of rectangles
    in the form [left, top, width, height]. This function
    takes that, and returnes (topleft, bottomright)"""
    x, y, width, height = rect
    topleft = (x, y)
    bottomright = (x + width, y + height)
    return topleft, bottomright

print("face recognition:    %f", timeit.timeit(fr, number=30))
print("opencv: %f" % timeit.timeit(cv, number=30))
print("dlib: %f" % timeit.timeit(dl, number=30))

frLocations = fr()
frimage = image.copy()
for frrect in frLocations:
    topleft, bottomright = faceCorners(frrect)
    cv2.rectangle(frimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('frimage', frimage)

cvLocations = cv()
cvimage = image.copy()
for cvrect in cvLocations:
    topleft, bottomright = cv2Corners(cvrect)
    cv2.rectangle(cvimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('cvimage', cvimage)

dlLocations, _, _ = dl()
dlimage = image.copy()
for dlrect in dlLocations:
    topleft, bottomright = dlibCorners(dlrect)
    cv2.rectangle(dlimage, topleft, bottomright, (0,0,0), 2)
cv2.imshow('dlimage', dlimage)

cv2.waitKey(0)