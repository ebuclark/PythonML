import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTreeClassifier

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
    )
    
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    print(clf.predict(X_test))
    print(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))
    clf.printTree()