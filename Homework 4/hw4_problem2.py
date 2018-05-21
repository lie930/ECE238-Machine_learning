import math
import numpy as np
import matplotlib.pyplot as plt
from functions.utils import generate_class_0, generate_class_1
from functions.adaboost import adaboost_def_M,adaboost_undef_M

if __name__ == '__main__':
    N = 200;

    # Generate data
    x0 = generate_class_0(math.floor(N/2))
    x1 = generate_class_1(math.floor(N/2))
    X = np.concatenate([x0, x1])
    Y = np.append(np.zeros(int(N/2)), np.ones(int(N/2)))

    # Shuffle the data
    ind_perm = np.random.permutation(len(Y))
    X = X[ind_perm, :]
    Y = Y[ind_perm]

    # Divide data into train- and test set
    n_train = int(np.floor(0.75 * N))
    X_train = X[0:n_train,:]
    Y_train = Y[0:n_train]
    n_test = int(np.floor(0.25 * N))
    X_test = X[n_train+1:n_train+n_test,:]
    Y_test = Y[n_train+1:n_train+n_test]

    M = 5
    adaboost_def_M(X_train, Y_train, X_test, Y_test, M)
    adaboost_undef_M(X_train, Y_train, X_test, Y_test)
    plt.show()
