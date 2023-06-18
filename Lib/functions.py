import numpy as np
import cv2
import time
import dlib
from math import gcd

class Default:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_speed = 0

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')

    def putText(frame, text, x, y):
        cv2.putText(frame, text, (x, y), Default.font, 0.5, (255, 0, 255), 1)
        
def pushback(lst, elem):
    lst = lst[1:]
    lst.append(elem)
    return lst

class FrameRate:
    def __init__(self, avg : int = 2):
        import time

        self.start = time.time()
        self._fps = [0] * avg

    def Update(self):
        now = time.time()
        time_diff = now - self.start
        newfps = 1 / time_diff
        self._fps = pushback(self._fps, newfps)

        self.start = now

    def getFPS(self):
        return min(self._fps)
        # return sum(self._fps) / len(self._fps)

    def showFPS(self, frame):
        self.Update()
        fps_text = "FPS: {:.1f}".format(self.getFPS())
        cv2.putText(frame, fps_text, (10, 20), Default.font, 0.5, (255, 0, 255), 1)
        # print(self._fps, end=' ' * 10 + '\r')

def getImage(img : cv2.Mat, topleft : tuple[2] | list[2], bottomright : tuple[2] | list[2]) -> cv2.Mat:
    """returns the image section from the top left corner to the bottom left corner
    takes (left, top), (right, bottom) and returns cv2.Mat"""
    left, top = topleft
    right, bottom = bottomright
    return img[top:bottom, left:right]
def addImage(base_image, overlay_image, x, y):
    overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

    base_height, base_width, _ = base_image.shape
    overlay_height, overlay_width, _ = overlay_image_rgb.shape

    if x + overlay_width > base_width or y + overlay_height > base_height:
        print('Error: Overlay image exceeds base image dimensions at given position')
        return base_image

    base_image[y:y+overlay_height, x:x+overlay_width] = overlay_image_rgb
    return base_image

def landmarksToPoints(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords
def rectFromLandmarks(points):
    top = 0
    left = 0
    bottom = 999
    right = 999
    for (x, y) in points:
        top = max(top, y)
        bottom = min(bottom, y)
        left = max(left, x)
        right = min(right, x)
    return (left, top), (right, bottom)

def imageAspectRatioGCD(image : cv2.Mat) -> tuple[2]:
    #800x600
    """takes in an image and returns the lowest possible aspect ratio"""
    height, width, _ = image.shape
    denominator = gcd(width, height)
    return (width//denominator, height//denominator)
def scaleImageMaxTo(image : cv2.Mat, maxLength : int) -> cv2.Mat:
    height, width, _ = image.shape
    longest = max(height, width)
    shortWidth = width/longest
    shortHeight = height/longest
    scaledSize = (int(shortWidth * maxLength), int(shortHeight * maxLength))
    return cv2.resize(image, scaledSize)

# def faceCorners(rect) -> tuple[2]:
#     topleft, bottomright = (rect[3], rect[0]), (rect[2], rect[1])
#     return topleft, bottomright
def dlibCorners(rect) -> tuple[2]:
    """takes a rect returned from dlib and returns 
    the top left corner, and the bottom right corner"""
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (left, top), (right, bottom)
def cv2Corners(rect) -> tuple[2]:
    """cv2's detectMultiScale return a list of rectangles
    in the form [left, top, width, height]. This function
    takes that, and returnes (topleft, bottomright)"""
    x, y, width, height = rect
    topleft = (x, y)
    bottomright = (x + width, y + height)
    return topleft, bottomright

if __name__ == "__main__":
    image = cv2.imread('../group.png')
    image = scaleImageMaxTo(image, 800)
    cv2.imshow('image', image)
    cv2.waitKey(0)