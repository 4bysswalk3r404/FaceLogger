import numpy as np
import cv2
import time
import dlib

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

def getImageRect(img: cv2.Mat, rect) -> cv2.Mat:
    x = rect.left()
    y = rect.top()
    return getImage(img, x, y, rect.right()-x, rect.bottom()-y)
def getImageTup(img: cv2.Mat, tup) -> cv2.Mat:
    return getImage(img, tup[0], tup[1], tup[2], tup[3])
def getImage(img: cv2.Mat ,x,y,w,h) -> cv2.Mat:
    return img[y:y+h, x:x+w]

def addImage(base_image, overlay_image, x, y):
    overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

    base_height, base_width, _ = base_image.shape
    overlay_height, overlay_width, _ = overlay_image_rgb.shape

    if x + overlay_width > base_width or y + overlay_height > base_height:
        print('Error: Overlay image exceeds base image dimensions at given position')
        return base_image

    base_image[y:y+overlay_height, x:x+overlay_width] = overlay_image_rgb
    return base_image

def rect_to_corners(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()
	return (x1, y1), (x2, y2)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def getRect(points):
    top = 0
    left = 0
    bottom = 999
    right = 999
    for (x, y) in points:
        if y > top: top = y
        if y < bottom: bottom = y
        if x > left: left = x
        if x < right: right = x
    return (left, top), (right, bottom)

def getImageCorners(img: cv2.Mat, topLeft : tuple, bottomRight : tuple) -> cv2.Mat:
    """x1, y1, x2, y2"""
    w = bottomRight[0] - topLeft[0]
    h = bottomRight[1] - topLeft[1]
    return getImage(img, topLeft[0], topLeft[1], w, h)