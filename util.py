import numpy as np
import os
import cv2
import glob
import toml

# returns settings dictionary
# must be in same dir as settings.toml to work
def loadSettings():
	f = open("settings.toml")
	settings_content = f.read()
	f.close()
	return toml.loads(settings_content)

# returns 2 4d numpy array
# (input, answer)
def loadTrainingData(pathToTrainingData):
    x_data = []

# converts a text file to a numpy array
def loadTextFile(path):
    return

# ignore this
def loadImageSet(folderPath):
    x_data = []
    files  = glob.glob(folderPath + "*.jpg")

    for file in files:
        image = cv2.imread(file)
        x_data.append(image)
    return np.array(x_data)