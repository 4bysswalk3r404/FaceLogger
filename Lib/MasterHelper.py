import cv2
import dlib
import face_recognition

from Lib.functions import getImageRect
from Lib.endpng import putInfo, getInfo, getOffset

import os
import json

detector = dlib.get_frontal_face_detector()

facefolder = "./Faces"
flagfile = os.path.join(facefolder, ".FLAG.json")

def rect_to_tup(rect):
    return (rect.left(), rect.top(), rect.right(), rect.bottom())

def addFace(filename, name):
    if not type(filename) is cv2.Mat:
        img = cv2.imread(filename)
    else:
        img = filename

    name = os.path.splitext(name)[0]

    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    locations, score, _ = detector.run(gray, 1)
    locations = locations[0]
    score = round(score[0], 4)

    encoding = face_recognition.face_encodings(small_frame, (rect_to_tup(locations), ))[0]

    readencoding = (str(encoding).replace(']', '').replace('[', '')).split()
    readencoding = [float(i) for i in readencoding]
    metadata = '{"Score" : %s, "Encoding": %s}' % (score, readencoding)

    faceimg = getImageRect(img, locations)

    cv2.imwrite(f"./Faces/{name}.png", faceimg)
    putInfo(f"./Faces/{name}.png", metadata)

    flag(f"{name}.png")

def updateEncoding(filename, encoding):
    metadata = getInfo(filename).decode()
    metadata = json.loads(metadata)
    metadata['Encoding'] = encoding

    offset = getOffset(filename)
    imsize = os.path.getsize(filename)

    img = open(filename, 'rb').read(imsize - offset)
    with open(filename, 'wb') as file:
        file.write(img)
        file.write(metadata.encode())
def getEncoding(name):
    if not os.path.exists(name):
        name = os.path.join(facefolder, name)
        if not os.path.exists(name):
            name.path.join(name, '.png')
    metadata = getInfo(name).decode()
    metadata = json.loads(metadata)
    return metadata['Encoding']

def updateScore(filename, score):
    metadata = getInfo(filename).decode()
    metadata = json.loads(metadata)
    metadata['Score'] = score

    offset = getOffset(filename)
    imsize = os.path.getsize(filename)

    img = open(filename, 'rb').read(imsize - offset)
    with open(filename, 'wb') as file:
        file.write(img)
        file.write(metadata.encode())

def updateFace():
    pass

def flag(itemname):
    data = json.load(open(flagfile))
    files = data['updated']

    if itemname in files:
        return
    data['updated'].append(itemname)
    json.dump(data, open(flagfile, 'w'))
def flagclear(args = None):
    if args != None:
        data = json.load(open(flagfile))
        data[args] = []
        json.dump(data, open(flagfile, 'w'))    
    else:
        json.dump({"updated" : [], "name" : [[],]}, open(flagfile, 'w'))    

def changeNameFlag(oldname, newname):
    data = json.load(open(flagfile))
    data["name"].append([oldname, newname])
    json.dump(data, open(flagfile, 'w'))

def getmetadata(filename):
    metadata = getInfo(filename).decode()
    metadata = json.loads(metadata)
    return metadata
