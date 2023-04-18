import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from DecisionTree import DecisionTree

from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    y = np.random.normal(10, 5, len(y))

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
    )
    
    # clf = DecisionTree(max_depth=10)
    # clf.fit(X_train, y_train)

    # print(clf.predict(X_test))
    # print(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))
    # clf.printTree()

    reg = DecisionTree(max_depth=10, mode='regression')
    reg.fit(X_train, y_train)

    dtr = DecisionTreeRegressor(max_depth=10, min_samples_split=2)
    dtr.fit(X_train, y_train)
    print(reg.predict(X_test))
    print(mean_squared_error(y_pred=reg.predict(X_test), y_true=y_test))
    print(mean_squared_error(y_pred=dtr.predict(X_test), y_true=y_test))