import math
import numpy as np
import matplotlib.pyplot as plt

from functions.utils import generate_class_0, generate_class_1
from functions.adaboost import adaboost

if __name__ == '__main__':
    N = 100;
    # Generate data
    x0 = generate_class_0(math.floor(N/2))
    x1 = generate_class_1(math.floor(N/2))
    x_train = np.concatenate([x0, x1])
    y = np.append(-1*np.ones(int(N/2)), np.ones(int(N/2)))
