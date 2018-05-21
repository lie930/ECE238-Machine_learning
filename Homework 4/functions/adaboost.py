from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Algorithm implementation based on notation from paper "Multi-class Adaboost (Zhu et al)"
def adaboost_def_M(X_train, Y_train, X_test, Y_test, M):
    n_train = len(X_train)
    n_test = len(X_test)
    w = np.ones(n_train) / n_train
    a = np.zeros(M)
    C = np.zeros((n_train, 2))
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    pred_test = np.zeros(n_test)
    Z_tot = np.zeros((n_train*n_train, 2))
    fig, axes = plt.subplots(1, M, figsize=(16,4))
    fig.tight_layout()
    fig.suptitle("Individual decision boundaries")
    X = np.linspace(-5, 7, n_train)
    Y = np.linspace(-5, 7, n_train)
    X, Y = np.meshgrid(X, Y)
    XY = np.vstack([ X.reshape(-1), Y.reshape(-1) ]).T

    for m in range(M):
        # Fit a classifier T to the training data using weights w
        clf_tree.fit(X_train, Y_train, sample_weight=w)
        prediction_m = clf_tree.predict(X_train)

        # Compute the error
        I = [int(x) for x in (prediction_m != Y_train)]
        err_m = (np.dot(w, I)) / np.sum(w)

        # Compute a
        a[m] = np.log( (1 - err_m) / float(err_m) )

        # Update weights
        w = np.multiply(w, np.exp([a[m] * float(x) for x in I]))

        # Renormalize weights
        w = w / np.sum(w)

        # Output
        prediction_m_onehot = np.zeros((n_train, 2))
        for i in range(n_train):
            prediction_m_onehot[i, int(prediction_m[i])] = 1

        C_m = a[m] * prediction_m_onehot
        C_m = np.argmax(C_m, axis=1).astype(np.float32)

        # Plot individual decision boundary
        prediction_grid = clf_tree.predict(XY)
        prediction_grid_onehot = np.zeros((n_train*n_train, 2))
        for i in range(n_train*n_train):
            prediction_grid_onehot[i, int(prediction_grid[i])] = 1

        Z = a[m] * prediction_grid_onehot
        Z_tot += Z
        Z = np.argmax(Z, axis=1).astype(np.float32)
        Z = Z.reshape((n_train, n_train))

        if not np.all(Z):
            Z[Z == 0] = -1
            axes[m].contour(X, Y, Z, 0)
        axes[m].scatter(X_train[:, 0], X_train[:, 1], c=C_m, cmap='Dark2', s=5)
        axes[m].set_title("m = %s"%(m+1))
        axes[m].set_aspect('equal')
        axes[m].axis([-5, 7, -5, 7])

    # Calculate missclassification
    pred_test_i = clf_tree.predict(X_test)
    pred_test = [sum(x) for x in zip(pred_test,[x * a[m] for x in pred_test_i])]
    pred_test = np.sign(pred_test)
    missclass = get_error_rate(pred_test, Y_test)

    # Plot overall decision boundary
    Z_tot = np.argmax(Z_tot, axis=1).astype(np.float32)
    Z_tot = Z_tot.reshape((n_train, n_train))
    Z_tot[Z_tot == 0] = -1

    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='Set1', s=10)
    plt.contour(X, Y, Z_tot, 0)
    plt.title("Overall decision boundary with M = %s"%M)

    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='Set1', s=10)
    plt.contour(X, Y, Z_tot, 0)
    plt.title("Decision boundary with test data, missclassification = %s"%np.round(missclass,3))

def adaboost_undef_M(X_train, Y_train, X_test, Y_test):
    n_train = len(X_train)
    n_test = len(X_test)
    w = np.ones(n_train) / n_train
    C = np.zeros((n_train, 2))
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    pred_test = np.zeros(n_test)
    Z_tot = np.zeros((n_train*n_train, 2))
    X = np.linspace(-5, 7, n_train)
    Y = np.linspace(-5, 7, n_train)
    X, Y = np.meshgrid(X, Y)
    XY = np.vstack([ X.reshape(-1), Y.reshape(-1) ]).T
    err_m = np.inf
    M = 0;
    while(True):
        # Fit a classifier T to the training data using weights w
        clf_tree.fit(X_train, Y_train, sample_weight=w)
        prediction_m = clf_tree.predict(X_train)

        # Compute the error
        I = [int(x) for x in (prediction_m != Y_train)]
        err_m = (np.dot(w, I)) / np.sum(w)

        # Compute a
        a_m = np.log( (1 - err_m) / float(err_m) )

        # Update weights
        w = np.multiply(w, np.exp([a_m * float(x) for x in I]))

        # Renormalize weights
        w = w / np.sum(w)

        # Output
        prediction_m_onehot = np.zeros((n_train, 2))
        for i in range(n_train):
            prediction_m_onehot[i, int(prediction_m[i])] = 1
        C_m = a_m * prediction_m_onehot
        C += C_m
        C_m = np.argmax(C_m, axis=1).astype(np.float32)

        # Plot individual decision boundary
        prediction_grid = clf_tree.predict(XY)
        prediction_grid_onehot = np.zeros((n_train*n_train, 2))
        for i in range(n_train*n_train):
            prediction_grid_onehot[i, int(prediction_grid[i])] = 1

        Z = a_m * prediction_grid_onehot
        Z_tot += Z
        Z = np.argmax(Z, axis=1).astype(np.float32)
        Z = Z.reshape((n_train, n_train))

        if np.all(Y_train == np.argmax(C, axis=1).astype(np.float32)):
            break
        else:
            M += 1

    # Calculate missclassification
    pred_test_i = clf_tree.predict(X_test)
    pred_test = [sum(x) for x in zip(pred_test,[x * a_m for x in pred_test_i])]
    pred_test = np.sign(pred_test)
    missclass = get_error_rate(pred_test, Y_test)


    # Plot overall decision boundary
    Z_tot = np.argmax(Z_tot, axis=1).astype(np.float32)
    Z_tot = Z_tot.reshape((n_train, n_train))
    Z_tot[Z_tot == 0] = -1

    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='Set1', s=10)
    plt.contour(X, Y, Z_tot, 0)
    plt.title("Overall decision boundary with M = %s"%M)

    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='Set1', s=10)
    plt.contour(X, Y, Z_tot, 0)
    plt.title("Decision boundary with test data, missclassification = %s"%np.round(missclass,3))


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))
