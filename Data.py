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
    rootnode = t.Node(array)
    ds.printData(data)
    labels = ds.getLabels(data)

    #print(array[0])
    tree = t.Tree('Gini')
    #q1 = t.Question(1, 5, labels)

    #rows = q1.class_counts(data)

    #true_rows = rows.pop()
    #print(q1.class_counts(rootnode.dataset))
    #print(q.match(array[1]))
    #true_row1, false_row1 = tree.partition(rootnode.dataset, q1)
    #control = t.Question(4, 'R', labels)
    #print(control.match(true_row1))
    #lchild1 = t.Node(true_row1)
    #q2 = t.Question(1, 1, labels)
    #true_row2, false_row2 = tree.partition(rootnode.dataset, q1)
    #rchild1 = t.Node(false_row1)


    gain, question = tree.find_split(rootnode.dataset, labels)

    print(question)
    #print(tree.entropy(true_row1, false_row1))
    print(tree.gini(rootnode.dataset))
    print(gain)



    #rootnode.left_child = lchild1
    #rootnode.right_child = rchild1
    #print(len(labels) -1)

    #print(true_row)

