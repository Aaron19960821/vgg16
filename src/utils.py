#!coding=utf-8

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
        preImage = cv2.imread(image)
        postImage = cv2.resize(preImage, (width, height), interpolation=cv2.INTER_CUBIC)
        res.append(res)

    return np.array(res)

def getLabels(trainSet, imageSet, labels):
    res = []
    for image in imageSet:
        res.append(labels[trainSet.index(image)])
    return res
