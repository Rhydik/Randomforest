import numpy as np

class RootNode:
    def __init__(self):
        return

class Tree:
    def __init__(self, dataset):
        self.root = None
        self.dataset = dataset

    def train(self):
        return

    def gini(self, p):
        return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))

    def entropy(self, p):
        return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

