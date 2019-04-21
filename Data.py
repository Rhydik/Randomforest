import pandas as pd
import numpy as np
import random as r


def get_features(data):
    features = list(data)
    myarray = np.asarray(features)
    return myarray


def get_labels(data):
    values = []
    for item in data:
        if not (values.__contains__(item)):
            values.append(item)
    return values


def print_data(dataset):
    print(dataset)


def load_data(dataset):
    data = pd.read_csv(dataset)
    return data


def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = r.randrange(len(dataset))
        sample.append(dataset[index])
    return sample
