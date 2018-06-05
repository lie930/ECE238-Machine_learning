from sklearn import linear_model
import numpy as np
import scipy.io

def lasso():
    lasso_values = scipy.io.loadmat('lasso_values.mat')
    matrix = lasso_values['lasso_mat']
    y = lasso_values['y']
    l = lasso_values['lambda']

    clf = linear_model.Lasso(alpha=l) # Set lambda ( called ’alpha ’ here )
    clf.fit(matrix,y) # Solve Lasso problem
    a_hat = clf.coef_ # Get a_hat

    dict = {
        'a_hat': a_hat
    }
    scipy.io.savemat('lasso_result',dict)
