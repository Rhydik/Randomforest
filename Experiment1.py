import Data as Data
import Tree as Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time as time
from Tree import calculate_accuracy_precision_recall_auc
import numpy as numpy
from sklearn.model_selection import train_test_split


# Data preparation
data = Data.load_data("Datasets/binary/heart-s.csv")

features = Data.get_features(data)

training_data = data.values.tolist()

# My classifiers
dt = Tree.Tree('Gini', 4, 100, None, features)
rf = Tree.RandomForest('Gini', 4, 20, None, 101, 1, None, features)

# Scikit-learn classifiers
rfc = RandomForestClassifier(criterion='gini', max_features=4, max_depth=20, n_estimators=101)
dtc = DecisionTreeClassifier(criterion="gini", max_features=4, max_depth=100)

# Cross Validation
train_data = numpy.array(data)
x_data = train_data[1:, :-1]

y_data = train_data[1:, -1:]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=100)

# Bootstrap data set
testing_data = Data.subsample(training_data)

# Results
# Fit my Decision tree
start = time.time()
my_tree = dt.fit(training_data)
end = time.time()
my_train_time = end - start

# Fit their Decision tree
start = time.time()
their_tree = dtc.fit(x_train, y_train)
end = time.time()
their_train_time = end - start

# Print my tree
# dt.print_tree(my_tree)

# Prediction my tree
start = time.time()
true_labels, my_pred = dt.predict_proba(testing_data, my_tree)
end = time.time()
my_test_time = end - start

# Prediction their tree
start = time.time()
their_pred = list(dtc.predict(x_test))
end = time.time()
their_test_time = end - start

# Stats my tree
my_accuracy, my_precision, my_recall, my_auc = calculate_accuracy_precision_recall_auc(true_labels, my_pred)

# Stats their tree
their_accuracy, their_precision, their_recall, their_auc = calculate_accuracy_precision_recall_auc(true_labels,
                                                                                                   their_pred)


# Accuracy Decision Tree
print(("Total accuracy for my Decision Tree: %.2f" % my_accuracy).replace(".", ","))
print(("Total accuracy for sklearn Decision Tree: %.2f" % their_accuracy).replace(".", ","))

# Precision Decision Tree
print(("\nTotal precision for my Decision Tree: %.2f" % my_precision).replace(".", ","))
print(("Total precision for sklearn Decision Tree: %.2f" % their_precision).replace(".", ","))

# Recall Decision Tree
print(("\nTotal recall for my Decision Tree: %.2f" % my_recall).replace(".", ","))
print(("Total recall for sklearn Decision Tree: %.2f" % their_recall).replace(".", ","))

# AUC Decision Tree
print(("\nTotal AUC for my Decision Tree: %.2f" % my_auc).replace(".", ","))
print(("Total AUC for sklearn Decision Tree: %.2f" % their_auc).replace(".", ","))

# Training Time Decision Tree
print(("\nTotal training time for my Decision Tree: %f" % my_train_time).replace(".", ","))
print(("Total training time for sklearn Decision Tree: %f" % their_train_time).replace(".", ","))

# Testing Time Decision Tree
print(("\nTotal testing time for my Decision Tree: %f" % my_test_time).replace(".", ","))
print(("Total testing time for sklearn Decision Tree: %f" % their_test_time).replace(".", ","))

# Fit my Random Forest
start = time.time()
my_trees = rf.fit(training_data)
end = time.time()
my_rf_train_time = end - start

# Fit their Random Forest
start = time.time()
their_trees = rfc.fit(x_train, y_train.ravel())
end = time.time()
their_rf_train_time = end - start

# Predict my Random Forest
start = time.time()
rf_true_labels, my_preds = rf.predict_proba(testing_data, my_trees)
end = time.time()
my_rf_test_time = end - start

# Predict their Random Forest
start = time.time()
their_preds = list(rfc.predict(x_test))
end = time.time()
their_rf_test_time = end - start

# Stats my Random Forest
my_rf_accuracy, my_rf_precision, my_rf_recall, my_rf_auc = calculate_accuracy_precision_recall_auc(rf_true_labels,
                                                                                                   my_preds)

# Stats their Random Forest
their_rf_accuracy, their_rf_precision, their_rf_recall, their_rf_auc = calculate_accuracy_precision_recall_auc(
    rf_true_labels, their_preds)

# Accuracy my Random Forest
print(("\n\nTotal accuracy for my Random Forest: %.2f" % my_rf_accuracy).replace(".", ","))
print(("Total accuracy for sklearn Random forest: %.2f" % their_rf_accuracy).replace(".", ","))

# Precision my Random Forest
print(("\nTotal precision for my Random Forest: %.2f" % my_rf_precision).replace(".", ","))
print(("Total precision for sklearn Random forest: %.2f" % their_rf_precision).replace(".", ","))

# Recall my Random Forest
print(("\nTotal recall for my Random Forest: %.2f" % my_rf_recall).replace(".", ","))
print(("Total recall for sklearn Random forest: %.2f" % their_rf_recall).replace(".", ","))

# AUC my Random Forest
print(("\nTotal AUC for my Random Forest: %.2f" % my_rf_auc).replace(".", ","))
print(("Total AUC for sklearn Random forest: %.2f" % their_rf_auc).replace(".", ","))

# Training Time my Random Forest
print(("\nTotal training time for my Random Forest: %f" % my_rf_train_time).replace(".", ","))
print(("Total training time for sklearn Decision Tree: %f" % their_rf_train_time).replace(".", ","))

# Testing Time my Random Forest
print(("\nTotal testing time for my Random Forest: %f" % my_rf_test_time).replace(".", ","))
print(("Total testing time for sklearn Random Forest Tree: %f" % their_rf_train_time).replace(".", ","))
