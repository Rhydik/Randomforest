import Data as d
import Tree as t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':

    #Data preparation
    data = d.loadData("Datasets/binary/balance-scale.csv")

    labels = d.getLabels(data)

    array = data.values.tolist()

    #My classifiers
    dt = t.Tree('Gini', 5, 100, None, labels)
    rf = t.Random_Forest('Gini', 5, 20, None, 100, 1, None, labels)

    #Scikit-learn classifiers
    rfc = RandomForestClassifier(n_estimators=5)
    dtc = DecisionTreeClassifier()

    my_tree = dt.fit(array)
    my_trees = rf.fit(array)


    #Bootstrap Dataset
    testing_data = d.subsample(array)
    #Results

    dt.print_tree(my_tree)
    print("Total accuracy for my Decision Tree: %.2f" % dt.predict_proba(testing_data, my_tree))

    print("Total accuracy for my Random Forest: %.2f" % (rf.predict_proba(testing_data, my_trees) / rf.n_estimators))
