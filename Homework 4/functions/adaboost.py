from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Algorithm implementation based on notation from paper "Multi-class Adaboost (Zhu et al)"

def adaboost(X_train, Y_train, X_test, Y_test, M, clf_tree):
    n_train = len(X_train)
    n_test = len(X_test)

    #Initiate weights
    w = np.ones(n_train)/n_train

    T = np.zeros((M,n_train))
    I = np.zeros((M,n_train))
    err = np.zeros(M)
    a = np.zeros(M)

    for m in range(M):
        # Fit a classifier T to the training data using weights w
        clf_tree.fit(X_train, Y_train, sample_weight = w)
        T[:][m] = clf_tree.predict(X_train)

        # Compute the error
        I[m] = [int(x) for x in (T[m] != Y_train)]
        err[m] = np.sum(np.dot(w,I[m])) / sum(w)

        # Compute alpha
        a[m] = np.log((1-err[m])/float(err[m]))

        # Update weights
        for i in range(n_train):
            w[i] = w[i] * np.exp(a[m]*I[m][i])

    #Return classifier C
    sum1 = np.zeros((n_train,2))
    k = [-1,1]
    for i in range(2):
        ki = k[i]
        for m in range(M):
            print(a[m])
            print([float(x) for x in (T[m] == ki)])
            #sum1[:][i] += a[m]*[int(x) for x in (T[m] == ki)]
'''
    for i in range(n_train):
        if sum1[i][1] > sum1[i][2]:
            C[i] = -1
        else:
            C[i] = 1
    return C
'''
