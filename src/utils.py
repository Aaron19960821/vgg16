import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import tensorflow as tf
import numpy as np
import json
import cv2
import random

def readJsonFromFile(filename):
    return json.load(open(filename))

def writeJsonToFile(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, filename)

def randomChoose(imageSet, number):
    random.shuffle(imageSet)
    return imageSet[:number]
    
def loadImagesFromFile(imageSet, width, height):
    res = []
    for image in imageSet:
        im1 = cv2.imread(image.strip('\n'))
        im2 = cv2.resize(im1, (width, height), interpolation=cv2.INTER_CUBIC)
        res.append(im2)

    return np.array(res)

def getLabels(trainSet, imageSet, labels):
    res = []
    for image in imageSet:
        res.append(labels[trainSet.index(image)])
    return res
