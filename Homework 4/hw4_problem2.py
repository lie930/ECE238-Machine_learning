import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from functions.utils import generate_class_0, generate_class_1
from functions.adaboost import adaboost

if __name__ == '__main__':
    N = 10;
    # Generate data
    x0 = generate_class_0(math.floor(N/2))
    x1 = generate_class_1(math.floor(N/2))
    X_train = np.concatenate([x0, x1])
    Y_train = np.append(-1*np.ones(int(N/2)), np.ones(int(N/2)))

    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    M = 1
    C = adaboost(X_train, Y_train, X_train, Y_train, M, clf_tree)
