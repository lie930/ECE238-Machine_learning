import random
import math
import numpy as np
import matplotlib.pyplot as plt

from functions.utils import generate_class_0, generate_class_1
from functions.SVM import SMOModel, gaussian_kernel, decision_function, train, plot_decision_boundary

N = 100 # Number of samples

# Generate data
x0 = generate_class_0(math.floor(N/2))
x1 = generate_class_1(math.floor(N/2))
x_train = np.concatenate([x0, x1])
y = np.append(-1*np.ones(int(N/2)), np.ones(int(N/2)))

# Shuffle the data
ind_perm = np.random.permutation(len(y))
x_train = x_train[ind_perm, :]
y = y[ind_perm]

# Set model parameters and initial values
C = 1.0
m = len(x_train)
tol = 0.01 # error tolerance
eps = 0.01 # alpha tolerance
initial_alphas = np.zeros(m)
initial_b = 0.0

# Instantiate model
model = SMOModel(x_train, y, C, gaussian_kernel,initial_alphas, initial_b, np.zeros(m))

# Initialize error cache
initial_error = decision_function(model.alphas, model.y, model.kernel,model.X, model.X, model.b) - model.y
model.errors = initial_error

output = train(model)

# Support vectors have non-zero alphas
mask = model.alphas != 0.0
frac_support_vectors = len(model.X[:,0][mask]) / len(model.X[:,0])

print("Fraction of training data points that are support vectors:")
print(frac_support_vectors)

classified_points = [decision_function(model.alphas, model.y,
                               model.kernel, model.X,
                               np.array(x), model.b) for x in model.X]
classified_points = np.array(classified_points).reshape(len(model.X[:, 0]), 1)

classified_points[classified_points < 0] = -1
classified_points[classified_points > 0] = 1

misclassification_error = 0
for i in range(len(y)):
    if classified_points[i] != y[i]:
        misclassification_error += 1

print("Probability of incorrect classification:")
missclass = misclassification_error / len(y)
print(missclass)

# Plotting
fig, ax = plt.subplots(figsize=(16,16))
grid, ax = plot_decision_boundary(output, ax)
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.title("Fraction support vectors = " + str(frac_support_vectors) + "\nProbability of incorrect classification = " + str(missclass))
plt.show()
