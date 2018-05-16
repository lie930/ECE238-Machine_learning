import numpy as np
import random
import math

def generate_class_0(N):
    m = [0,0]
    C = [[2, 0],[0,1]]
    return np.random.multivariate_normal(m, C, N)

def generate_class_1(N):
    m_A = [-2,1]
    C_A = [[9.0/8, 7.0/8],[7.0/8, 9.0/8]]
    m_B = [3,2]
    C_B = [[2, 1],[1, 2]]

    pi_1 = 1.0/3

    x1 = np.zeros((N,2))
    for i in range(0, N):
        a = random.uniform(0, 1)
        if (a < pi_1):
            x1[i, :] = np.random.multivariate_normal(m_A, C_A, 1)
        else:
            x1[i, :] = np.random.multivariate_normal(m_B, C_B, 1)
    return x1
