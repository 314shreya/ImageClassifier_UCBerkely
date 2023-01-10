#Utility functions file 
import collections

import numpy as np
from dataClassifier import basicFeatureExtractorDigit, basicFeatureExtractorFace
import util
import samples
import matplotlib.pyplot as plt

def read_data(f):

    dict_images={}
    k=0

    lines=[line for line in open(f)]

    for i in range(0,len(lines),28):
        dict_images[k]=lines[i:i+28]
        k+=1

    return dict_images

def check( out):
    prob = dict(collections.Counter(out))
    for k in prob.keys():
        prob[k] = prob[k] / float(len(out))
    return prob

traindata='./digitdata/trainingimages'
#read_data(traindata)
trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 100)
rawTrainingData = samples.loadDataFile("facedata/facedatatrain", 100,60,70)
trainingData = list(map(basicFeatureExtractorFace, rawTrainingData))
# trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 1000)
# rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 1000,28,28)
# trainingData = list(map(basicFeatureExtractorDigit, rawTrainingData))

intial = dict(collections.Counter(trainingLabels))  # Get the number of training labels

for k in intial.keys():
    intial[k] = intial[k] / float(len(trainingLabels))

sec = dict()  # Intialize a dictionary for sec

for x, prob in intial.items():  # For every item we create a new dict
    sec[x] = collections.defaultdict(list) # Create the sec of default dictionary list

for x, prob in intial.items():
    first = list()
    for i, ptr in enumerate(trainingLabels):        # go through the traningLabels and check the indexs and append
        if x == ptr:                                # Check the index 
            first.append(i)

    second = list()

    for i in first:     # Second is list that will contain training data based on labels
        second.append(trainingData[i])

    for y in range(len(second)):    # Now we populate the dictionary with the correct label and the data
        for k, ptr in second[y].items():
            sec[x][k].append(ptr)

count = [a for a in intial] # Get the total count
for key, value in sec.items():
    #sec1=np.array(sec)
    print(key,len(value))
    for i in value:
        print(i)
for x in count:     
    for k, ptr in second[x].items():
        sec[x][k] =check(sec[x][k])