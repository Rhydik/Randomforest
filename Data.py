import pandas as pd
import Tree as t

class Dataset:
    def __init__(self):
        return

    def getLabels(self, data):
        labels = list(data)
        return labels

    def printData(self, dataset):
        print(dataset)

    def loadData(self, dataset):
        data = pd.read_csv(dataset)
        return data

if __name__ == '__main__':
    ds = Dataset()

    data = ds.loadData("Datasets/binary/balance-scale.csv")

    tree = t.Tree(data)
    ds.printData(data)
    labels = ds.getLabels(data)
    print(labels)

