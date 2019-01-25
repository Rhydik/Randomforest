import numpy as np
import Data as d

class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.left_child = None
        self.right_child = None

class Tree:
    def __init__(self):
        return


    def train(self):
        return

    def get_gini(self, p):
        return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))

    def get_entropy(self, p):
        return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

    def find_split(self, labels, dataset):
        for label in labels:
            return

    def partition(self, data, question):
        true_rows, false_rows = [], []
        for row in data:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):

        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self, labels):
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            labels[self.column], condition, str(self.value))

    def is_numeric(self, value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)