import pandas as pd
import Tree as t
import numpy as np

class Dataset:
    def __init__(self):
        return

    def getLabels(self, data):
        labels = list(data)
        myarray = np.asarray(labels)
        return myarray

    def printData(self, dataset):
        print(dataset)

    def loadData(self, dataset):
        data = pd.read_csv(dataset)
        return data

if __name__ == '__main__':
    ds = Dataset()

    data = ds.loadData("Datasets/binary/balance-scale.csv")


    array = data.values.tolist()
    rootnode = t.Node(data)
    ds.printData(data)
    labels = ds.getLabels(data)

    print(array[1])
    tree = t.Tree
    q = t.Question(0, 2)
    print(q.match(array[1]))
    true_row, false_row = tree.partition(tree, data.values.tolist(), q)

    print(true_row)

