# simple neural network for learning the xor function. There is one hidden layer with three nodes. Based on the following blog post:
#

import numpy as np
from numpy import random, zeros, maximum, sqrt, sum, any, array, abs, size
import unittest

# TODO: include a diagram of the network structure


# article idea: unit tests force one to work through the details. It forces you to confront failure. You have to fail to think more accurately.

# activation function
f = lambda x: maximum(0, x)

def evaluate_network(x, n):
	# n: tuple of tuple of matrices and biases for each layer in the network
	W, b1 = n[0]
	V, b2 = n[1]

	# evaluates the network
	return f(V.dot(f(W.dot(x) + b1)) + b2) 

def initialize_network():

	# initializes the network by returning randomly generated weight matrices with a normalized variance
	# as recommended by http://cs231n.github.io/neural-networks-2/.
	W = random.randn(3,2) * sqrt(2.0 / 2.0) # sqrt(2.0 / # of input nodes)
	V = random.randn(2, 3) * sqrt(2.0 / 3.0) # sqrt(2.0 / # of input nodes)

	# we set biases to zero.
	b1 = zeros(2)
	b2 = zeros(3)

	return (W, b1), (V, b2)

def loss_function(x, y, n):
	# computes the loss energy for a given network (choice of W, V). 
	# x: rows of feature vectors (inputs)
	# y: rows of classification vectors (outputs)

	err = 0
	# loop through the rows
	for i in range(size(x, 1)):
		err += sum(abs(evaluate_network(x[i, :], n) - y[i, :]))**2

	# Q: why do we normalize the l^2 error by 1 / 2m? Something to do with estimators?
	return 1.0 / (2.0 * size(x, 1)) * err

def gradient_descent(x0, grad, alpha, iterations):
	# gradient descent algorithm for finding the minimum of a function
	x = x0

	for i in range(iterations):
		x -= alpha * grad(x)
	return x

def evaluate_loss_gradient(x, y):
	pass

class EvaluateNetworkTestCase(unittest.TestCase):

	def test_gradient_descent(self):
		# find the minimum of f(x) = x^2
		grad = lambda x: 2.0 * x

		estMin = gradient_descent(.5**2, grad, .01, 300)

		self.assertAlmostEqual(estMin, 0.0, delta=.001)

	def test_loss_function(self):
		# usual network setup with only the top node turned on
		W = zeros((3,2))
		W[0,0] = 1 # turn on connection between input layer first node and hidden layer first node
		b1 = zeros(3)

		V = zeros((2, 3))
		b2 = zeros(2)

		V[0,0] = 1 # turn on connnection between hidden layer first node and first node output layer

		n = ((W, b1), (V, b2))

		# evaluates the gradient of the loss function for training data (x, y)
		x = array([[1, 0], [0, 0]])
		y = array([[1, 0], [0, 0]])

		loss = loss_function(x, y, n)

		# check type signature (sadly the compiler won't do it for us)
		self.assertEquals(type(loss), np.float64)

		# should be zero loss
		self.assertAlmostEqual(loss, 0, delta=.01)

		# now test the loss function where we actually have some loss
		x = array([[.5, 0], [0, 0]])
		y = array([[1, 0], [0, 0]])

		loss = loss_function(x, y, n)

		self.assertEquals(type(loss), np.float64)

		# will be (x - y)^2 * 1/2m where m is the number of datapoints. not completely clear why we normalize like this?
		self.assertAlmostEquals(loss, .5**2 * 1.0 / (2.0 * 2.0), delta=.001)

	def test_network_initialization(self):
		(W,b1), (V, b2) = initialize_network()
	

		# make sure weights not all zero (ignore the augmented)
		self.assertTrue(any(abs(W) > 0))
		self.assertTrue(any(abs(V) > 0))

		# compute the estimators for the variance and mean
		# Q: is sampling from on normal distribution many times the same as sampling once from a single normal distribution?
		# Q: what uniquely determines a distribution? Can you reconstruct it from the PMF? What about from the CDF? Moments?
		# Q: how do Gaussians add? Means and variances are additive?
		# Q: Why do we scale by 1/sqrt{n} where n is the number of input nodes
		# A: (partial) when we scale a random variable, the variance changes by the square Var(aX) = a^2 Var(X)
		# where X represents a sample from n dimensional Gaussian (this is the orientation of the hyperlane representing this vertex)
		# If we want to normalize the variance of X by 1/n then we should scale X by 1/sqrt{n}. 
		# Q: where does the 2 come from?
		# Q: are we changing the variance by scaling?

		# checking to see that the mean around each point is zero
		# The sampling average has variance \sigma^2 / n so since \sigma = 1, we expect errors within 1 / 

		# we put any sophisticated unit tests on hold for now since it requires more advanced knowledge of the distribution
		# of the estimators.
		# self.assertAlmostEqual(np.average(W), 0, delta=.05) # make sure we have sample mean roughly zero
		# self.assertAlmostEqual(np.average(V), 0, delta=.05) # make sure we have sample mean roughly zero

		# # checking to see that variance is sqrt(2/n)
		# # Using unbiased estimator 1/(n - 1) sum[ (x_i - mu)^2 ]
		# var1 = 1.0 / 5.0 * sum((W[:, 0:1] - mean1)**2)
		# var2 = 1.0 / 5.0 * sum((W[:, 0:1] - mean1)**2)

		# # we scale the random variable X by sqrt(2 / # of input nodes). Then the variance scales by 2 / # of input nodes.
		# self.assertAlmostEqual(var1, 2.0 / 3.0, delta=.05)
		# self.assertAlmostEqual(var2, 2.0 / 2.0, delta=.05)

	def test_activation_function(self):
		x = array([-1, 2, 0])

		self.assertTrue(all(f(x) == array([0, 2, 0])))

	def evaluate_pass_thru_network(self):
		# turn on first input should turn on first output

		# weights, biases for the first layer (note that we are using projective representation for the affine transformations)
		W = zeros((3,2))
		W[0,0] = 1 # turn on connection between input layer first node and hidden layer first node
		b1 = zeros(3)

		V = zeros((2, 3))
		b2 = zeros(2)

		V[0,0] = 1 # turn on connnection between hidden layer first node and first node output layer

		# note that our vector has an extra coordinate for the translate part of the affine transformation.
		# or, equivalently, think of it as an extra dummy node with weight given by the bias. Will be a dummy node at every layer?
		x = array([1, 0])


		self.assertItemsEqual(evaluate_network(x, ((W, b1), (V, b2))), array([1, 0]))

	def test_bias(self):
		# shift the signal up by +1 for the first node in the hidden layer. No bias for all the other nodes.
		W = zeros((3,2))
		W[0,0] = 1
		print(W)
		b1 = array([1, 0, 0])

		V = zeros((2, 3))
		V[0,0] = 1 # turn on connnection between hidden layer first node and first node output layer
		b2 = zeros(2)

		x = array([1, 0])

		self.assertItemsEqual(f(W.dot(x) + b1), array([2, 0, 0])) # eval first layer of the network

		self.assertItemsEqual(f(V.dot(f(W.dot(x) + b1)) + b2), array([2, 0])) # eval second layer of the network

		self.assertItemsEqual(evaluate_network(x, ((W, b1), (V, b2))), array([2, 0]))

if __name__ == "__main__":
	unittest.main()