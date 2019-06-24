import Data as Data
import Tree as Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Tree import calculate_accuracy_precision_recall_auc
import numpy as numpy
from sklearn.model_selection import train_test_split


def run():
    # Data preparation
    data = Data.load_data("Datasets/binary/tic-tac-toe.csv")
    features = Data.get_features(data)

    training_data = data.values.tolist()

    # Cross Validation
    train_data = numpy.array(data)
    x_data = train_data[1:, :-1]

    y_data = train_data[1:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=100)

    testing_data = Data.subsample(training_data)

    max_features_step = 2
    max_depth_step = 2

    algorithmResults = {}
    algorithmResults['DT'] = []
    algorithmResults['RF'] = []
    algorithmResults['DTC'] = []
    algorithmResults['RFC'] = []
    algorithmResults['KNN'] = []

    for x in range(3, 13):
        max_features = x * max_features_step
        algorithmResults['DT'].append([])
        algorithmResults['RF'].append([])
        algorithmResults['DTC'].append([])
        algorithmResults['RFC'].append([])
        algorithmResults['KNN'].append([])

        for y in range(1, 20):
            max_depth = y * max_depth_step
            dt = Tree.Tree(criterion='Gini', max_features=max_features, max_depth=max_depth, min_sample_leafs=4,
                           labels=features)
            rf = Tree.RandomForest(criterion='Gini', max_features=max_features, max_depth=max_depth, min_sample_leaf=4,
                                   n_estimators=101, bagging=True, sample_size=None, labels=features)
            rfc = RandomForestClassifier(criterion='gini', max_features=max_features, max_depth=max_depth,
                                         n_estimators=101)
            dtc = DecisionTreeClassifier(criterion="gini", max_features=max_features, max_depth=max_depth)
            knn = KNeighborsClassifier(n_neighbors=max_features, leaf_size=max_depth)

            # Results
            # Fit my Decision tree
            my_tree = dt.fit(training_data)

            # Fit their Decision tree
            their_tree = dtc.fit(x_train, y_train)

            # Prediction my tree
            true_labels, my_pred = dt.predict_proba(testing_data, my_tree)

            # Prediction their tree
            their_pred = list(dtc.predict(x_test))

            # Stats my tree
            my_accuracy = calculate_accuracy_precision_recall_auc(true_labels, my_pred)

            # Stats their tree
            their_accuracy = calculate_accuracy_precision_recall_auc(true_labels, their_pred)

            # Fit my Random forest
            my_trees = rf.fit(training_data)

            # Fit their Random Forest
            their_trees = rfc.fit(x_train, y_train.ravel())

            # Predict my Random Forest
            rf_true_labels, my_preds = rf.predict_proba(testing_data, my_trees)

            # Predict their Random Forest
            their_preds = list(rfc.predict(x_test))

            # Stats my Random Forest
            my_rf_accuracy = calculate_accuracy_precision_recall_auc(rf_true_labels, my_preds)

            # Stats their Random Forest
            their_rf_accuracy = calculate_accuracy_precision_recall_auc(rf_true_labels, their_preds)

            # Fit KNeighbors
            knn.fit(x_train, y_train)

            # Predict KNeighbors
            knn_pred = list(knn.predict(x_test))

            # Stats KNeighbors
            knn_accuracy = calculate_accuracy_precision_recall_auc(true_labels, their_pred)


def add_datasets():
    datasets = []
    datasets.append("Datasets/binary/balance-scale.csv")
    datasets.append("Datasets/binary/breast-cancer.csv")
    datasets.append("Datasets/binary/breast-w.csv")
    datasets.append("Datasets/binary/credit-a.csv")
    datasets.append("Datasets/binary/credit-g.csv")
    datasets.append("Datasets/binary/diabetes.csv")
    datasets.append("Datasets/binary/haberman.csv")
    datasets.append("Datasets/binary/heart-c.csv")
    datasets.append("Datasets/binary/heart-h.csv")
    datasets.append("Datasets/binary/heart-s.csv")
    datasets.append("Datasets/binary/hepatitis.csv")
    datasets.append("Datasets/binary/ionosphere.csv")
    datasets.append("Datasets/binary/kr-vs-kp.csv")
    datasets.append("Datasets/binary/labor.csv")
    datasets.append("Datasets/binary/liver-disorders.csv")
    datasets.append("Datasets/binary/mushrooms.csv")
    datasets.append("Datasets/binary/sick.csv")
    datasets.append("Datasets/binary/sonar.csv")
    datasets.append("Datasets/binary/spambase.csv")
    datasets.append("Datasets/binary/tic-tac-toe.csv")
    return datasets
