import pandas as pd
import Tree as t
import numpy as np



def getLabels(data):
    labels = list(data)
    myarray = np.asarray(labels)
    return myarray

def printData(dataset):
    print(dataset)

def loadData(dataset):
    data = pd.read_csv(dataset)
    return data

