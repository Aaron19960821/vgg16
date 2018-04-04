import tensorflow as tf
import numpy as np
import json

def readJsonFromFile(filename):
    return json.load(open(filename))

def writeJsonToFile(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, filename)
