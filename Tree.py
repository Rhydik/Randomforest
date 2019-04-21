import numpy as np
import Data as Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def auc_score(precision, recall):
    if recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


def class_counts(data):
    counts = {}
    for row in data:
        result = row[-1]
        if result not in counts:
            counts[result] = 0
        counts[result] += 1
    return counts


def calculate_accuracy_precision_recall_auc(label, data):
    if len(label) > len(data):
        n = len(label) - len(data)
        label = label[:-n]

    accuracy = accuracy_score(label, data) * 100
    precision = precision_score(label, data, average='macro') * 100
    recall = recall_score(label, data, average='macro') * 100
    auc = auc_score(precision, recall)

    return accuracy, precision, recall, auc


class Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


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

    def predict_proba(self, data, node):
        labels = []
        prediction = []

        for row in data:
            labels.append(row[-1])
            prediction.append(list(self.predict(row, node).keys()).pop())

        return labels, prediction

    def fit(self, data):
        gain, question = self.find_split(data)

        self.max_depth = self.max_depth - 1
        if gain == 0 or self.min_sample_leafs == 0 or self.max_depth == 0:
            return Leaf(data)

        true_rows, false_rows = self.partition(data, question)

        true_branch = self.fit(true_rows)

        false_branch = self.fit(false_rows)

        return Node(question, true_branch, false_branch)

    def find_split(self, data):
        best_gain = 0
        best_question = None
        n_features = self.max_features
        current_uncertainty = 0

        if self.criterion == 'Gini':
            current_uncertainty = self.gini(data)

        elif self.criterion == 'Entropy':
            current_uncertainty = self.entropy(data)

        if self.max_features is None:
            n_features = len(self.labels) - 1
        elif self.max_features > len(self.labels):
            n_features = len(self.labels) - 1

        for col in range(n_features - 1):
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

    @staticmethod
    def gini(data):
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

    @staticmethod
    def entropy(data):
        rows = class_counts(data)

        list_of_labels = list(rows.values())
        n_false_rows = float(list_of_labels.pop())
        n_true_rows = float(list_of_labels.pop())

        if n_false_rows == 0 or n_true_rows == 0:
            return 0
        else:
            return -(n_true_rows/(n_true_rows + n_false_rows)) * np.log2(n_true_rows/(n_true_rows + n_false_rows)) - \
                   (n_false_rows/(n_true_rows + n_false_rows)) * np.log2(n_false_rows/(n_true_rows + n_false_rows))

    @staticmethod
    def partition(data, question):
        true_rows, false_rows = [], []
        for row in data:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def info_gain(self, true_rows, false_rows, current_uncertainty):
        p = float(len(true_rows) / ((len(true_rows)) + (len(false_rows))))

        if self.criterion == 'Gini':
            return current_uncertainty - p * self.gini(true_rows) - (1 - p) * self.gini(false_rows)

        elif self.criterion == 'Entropy':
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

    @staticmethod
    def is_numeric(value):
        return isinstance(value, int) or isinstance(value, float)


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class RandomForest:
    def __init__(self, criterion, max_features, max_depth, min_sample_leaf, n_estimators, bagging, sample_size, labels):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.n_estimators = n_estimators
        self.bagging = bagging
        self.sample_size = sample_size
        self.labels = labels

    def fit(self, data):
        t = Tree(self.criterion, self.max_features, self.max_depth, self.min_sample_leaf, self.labels)

        trees = []

        for i in range(self.n_estimators):
            if i == self.n_estimators:
                return
            if self.bagging:
                data = Data.subsample(data)
            my_tree = t.fit(data)
            trees.append(my_tree)
        return trees

    def predict(self, data, trees):
        t = Tree(self.criterion, self.max_features, self.max_depth, self.min_sample_leaf, self.labels)
        predictions = []
        for tree in trees:

            predictions.append(t.predict(data, tree))

        return predictions

    def predict_proba(self, data, trees):
        t = Tree(self.criterion, self.max_features, self.max_depth, self.min_sample_leaf, self.labels)

        labels = []
        predictions = []

        for tree in trees:
            for row in data:
                labels.append(row[-1])
                predictions.append(list(t.predict(row, tree).keys()).pop())

        return labels, predictions
