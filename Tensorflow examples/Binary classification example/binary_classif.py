'''
Binary classification of digits 3 and 7 from the MNIST dataset
Uses a neural network with 1 hidden layer
'''

import tensorflow as tf
import numpy as np
from MNIST_data import read_mnist_pair

# Read data containing digits 3 and 7. The labels are either 0 or 1
mnist = read_mnist_pair.read_data_sets('./MNIST_data', digit1 = 3, digit2 = 7, test_ratio = 0.2, validation_ratio = 0.1)
N_train = mnist.train.images.shape[0]
data_shape = mnist.train.images.shape[1]

# Hyperparameters
n_hidden = 5 # Number of neurons in hidden layer
learning_rate = 0.01
reg_const = 0.01
batch_size = 100
N_epochs = 100

# Build model
x = tf.placeholder(tf.float32, shape=[None, data_shape])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal(shape=[data_shape, n_hidden]))
b1 = tf.Variable(tf.zeros(shape=[n_hidden]))
a1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w_out = tf.Variable(tf.random_normal(shape=[n_hidden, 1]))
b_out = tf.Variable(tf.zeros(shape=[1]))
logits = tf.matmul(a1, w_out) + b_out
output = tf.sigmoid(logits)

cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) # Binary cross entropy
reg_loss = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w_out) # L2 regularization
total_loss = tf.reduce_mean(cross_entropy_loss + reg_const*reg_loss) 

train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss)

prediction = tf.cast(tf.greater(output, 0.5), tf.float32) # Threshold output to 0 and 1 to get predicted labels
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32)) # Compare prediction to true labels

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Training
print("\nTraining:\n")
for epoch in range(N_epochs):
	for i in range(np.int(N_train/batch_size)):
		x_train_batch, y_train_batch = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={x:x_train_batch, y:y_train_batch})

	# Print training and validation accuracy every 10 epochs
	if epoch % 10 == 0:
		acc_train = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels})
		acc_validation = sess.run(accuracy, feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
		print("Epoch {:2d}, Train acc. {:.2f}, Validation acc. {:.2f}".format(epoch, 100*acc_train, 100*acc_validation))

# After training/ hyperparameter tuning is done, print test accuracy
acc_test = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
print("\nTest accuracy: {:.2f}".format(100*acc_test))