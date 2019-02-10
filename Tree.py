import numpy as np
import Data as d
import random as r


def class_counts(data):
    counts = {}
    for row in data:
        result = row[-1]
        if result not in counts:
            counts[result] = 0
        counts[result] += 1
    return counts

class Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

    def get_data_for_node(self):
        return self.dataset

class Tree:
    def __init__(self, criterion, max_features, max_depth, min_sample_leafs, labels):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leafs = min_sample_leafs
        self.labels = labels

    def predict(self, data, node):
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(data):
            return self.predict(data, node.true_branch)
        else:
            return self.predict(data, node.false_branch)

    def predict_proba(self, counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

    def accuracy_metric(self, actual, predicted):
        correct = 0


        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    def fit(self, data):
        gain, question = self.find_split(data)

        if gain == 0:
            return Leaf(data)

        true_rows, false_rows = self.partition(data, question)

        true_branch = self.fit(true_rows)

        false_branch = self.fit(false_rows)

        return Node(question, true_branch, false_branch)


    def find_split(self, data):
        best_gain = 0
        best_question = None

        if self.criterion == 'Gini':
            current_uncertainty = self.gini(data)

        elif self.criterion == 'Entropy':
            current_uncertainty = self.entropy(data)

        n_features = len(self.labels) - 1

        for col in range(n_features):
            values = set([row[col] for row in data])
            for val in values:
                question = Question(col, val, self.labels)

                true_rows, false_rows = self.partition(data, question)


                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = self.info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question

    def gini(self, data):
        rows = class_counts(data)

        list_of_labels = list(rows.values())

        if not list_of_labels:
            return 0
        else:
            n_false_rows = float(list_of_labels.pop())
            if not list_of_labels:
                return 0
            else:
                n_true_rows = float(list_of_labels.pop())



        if n_true_rows == 0 or n_false_rows == 0:
            return 0
        else:
            return 1 - (n_true_rows/(n_true_rows + n_false_rows))**2 - (n_false_rows/(n_true_rows + n_false_rows))**2


    def entropy(self, data):
        rows = class_counts(self, data)


        list_of_labels = list(rows.values())
        n_false_rows = float(list_of_labels.pop())
        n_true_rows = float(list_of_labels.pop())

        if n_false_rows == 0 or n_true_rows == 0:
            return 0
        else:
            return -(n_true_rows/(n_true_rows + n_false_rows)) * np.log2(n_true_rows/(n_true_rows + n_false_rows)) - \
                   (n_false_rows/(n_true_rows + n_false_rows)) * np.log2(n_false_rows/(n_true_rows + n_false_rows))

    def partition(self, data, question):
        true_rows, false_rows = [], []
        for row in data:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def info_gain(self, true_rows, false_rows, current_uncertainty):
        p = float(len(true_rows) / ((len(true_rows)) + (len(false_rows))))


        if (self.criterion == 'Gini'):
            return current_uncertainty - p * self.gini(true_rows) - (1 - p) * self.gini(false_rows)

        elif (self.criterion == 'Entropy'):
            return current_uncertainty - p * self.entropy(true_rows) - p * self.entropy(false_rows)

    def print_tree(self, node, spacing=""):

        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return

        print(spacing + str(node.question))

        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

class Question:
    def __init__(self, column, value, labels):
        self.column = column
        self.value = value
        self.labels = labels

    def match(self, example):

        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.labels[self.column], condition, str(self.value))

    def is_numeric(self, value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


if __name__ == '__main__':


    data = d.loadData("Datasets/binary/balance-scale.csv")

    labels = d.getLabels(data)

    array = data.values.tolist()

    t = Tree('Gini', None, None, None, labels)

    my_tree = t.fit(array)

    t.print_tree(my_tree)

    testing_data = (
        [3, 3, 2, 1, 'R'],
        [5, 1, 3, 4, 'L'],
        [1, 5, 2, 2, 'L'],
        [2, 2, 3, 5, 'R'],
        [1, 3, 3, 2, 'L'],
    )
    labels = []

    prediction = []
    for row in testing_data:
        print("Actual: %s. Predicted: %s" % (row[-1], t.predict_proba(t.predict(row, my_tree))))
        labels.append(row[-1])
        prediction.append(list(t.predict(row, my_tree).keys()))

    labelsarray = np.asarray(labels)

    predarray = np.asarray(prediction)


    print(t.accuracy_metric(labelsarray, predarray))

