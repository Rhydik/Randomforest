import numpy as np
import Data as d

class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.left_child = None
        self.right_child = None

    def get_data_for_node(self):
        return self.dataset

class Tree:
    def __init__(self, criterion):
        self.criterion = criterion


    def train(self):
        return

    def find_split(self, data, labels):
        best_gain = 0
        best_question = None

        if self.criterion == 'Gini':
            current_uncertainty = self.gini(data)

        elif self.criterion == 'Entropy':
            current_uncertainty = self.entropy(data)

        n_features = len(labels) - 1

        for col in range(n_features):
            values = set([row[col] for row in data])
            for val in values:
                question = Question(col, val, labels)

                true_rows, false_rows = self.partition(data, question)


                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = self.info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question

    def gini(self, data):
        rows = Question.class_counts(self, data)


        list_of_labels = list(rows.values())
        n_false_rows = float(list_of_labels.pop())
        n_true_rows = float(list_of_labels.pop())



        if n_true_rows == 0 or n_false_rows == 0:
            return 0
        else:
            return 1 - (n_true_rows/(n_true_rows + n_false_rows))**2 - (n_false_rows/(n_true_rows + n_false_rows))**2


    def entropy(self, true_rows, false_rows):
        n_true_rows = len(true_rows)
        n_false_rows = len(false_rows)

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
            return current_uncertainty

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

    def class_counts(self, data):
        counts = {}
        for row in data:
            result = row[-1]
            if result not in counts:
                counts[result] = 0
            counts[result] += 1
        return counts

    def __repr__(self):
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.labels[self.column], condition, str(self.value))

    def is_numeric(self, value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

