import pandas as pd
import Tree as t
import numpy as np
import random as r



def getLabels(data):
    labels = list(data)
    myarray = np.asarray(labels)
    return myarray

def printData(dataset):
    print(dataset)

def loadData(dataset):
    data = pd.read_csv(dataset)
    return data

def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = r.randrange(len(dataset))
		sample.append(dataset[index])
	return sample
