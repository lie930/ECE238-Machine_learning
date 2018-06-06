from sklearn import linear_model
import numpy as np
import scipy.io

def lasso():
    lasso_values = scipy.io.loadmat('lasso_values.mat')
    lasso_mat = lasso_values['lasso_mat']
    y = lasso_values['y']
    l = lasso_values['lambdas']
    N = len(y[0])


    n_lambdas = len(l[0])
    a_hat = np.zeros((N,6,n_lambdas))
    for i in range(n_lambdas):
        alpha = l[0][i]
        clf = linear_model.Lasso(alpha=alpha, max_iter=10000) # Set lambda ( called ’alpha ’ here )
        clf.fit(lasso_mat,y) # Solve Lasso problem
        a_hat[:,:,i] = clf.coef_ # Get a_hat
    dict = {
        'a_hat': a_hat
    }
    scipy.io.savemat('lasso_result',dict)
