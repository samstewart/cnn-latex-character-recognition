# simple neural network for learning the xor function. There is one hidden layer with three nodes. Based on the following blog post:
#

import numpy as np
import unittest

# TODO: include a diagram of the network structure


# article idea: unit tests force one to work through the details. It forces you to confront failure. You have to fail to think more accurately.

# activation function
f = lambda x: np.maximum(0, x)

def evaluate_network(x, n):
	# n: tuple of matrices for each layer of the network	
	W = n[0]
	V = n[1]

	# evaluates the network for a given point to classify.
	return f(V.dot(f(W.dot(x)))) # oooh, really nice notation. With biases b, c this is equivalent to f(V f(W x + b) + c)

def loss_function(x):
	# computes the loss energy for a given network (choice of W, V)
	pass

def evaluate_loss_gradient(x, y):
	# evaluates the gradient of the loss function for training data (x, y)
	pass

class EvaluateNetworkTestCase(unittest.TestCase):

	def test_activation_function(self):
		x = np.array([-1, 2, 0])

		self.assertTrue(np.all(f(x) == np.array([0, 2, 0])))

	def evaluate_pass_thru_network(self):
		# turn on first input should turn on first output

		# weights, biases for the first layer (note that we are using projective representation for the affine transformations)
		W = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])

		V = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

		# note that our vector has an extra coordinate for the translate part of the affine transformation.
		# or, equivalently, think of it as an extra dummy node with weight given by the bias. Will be a dummy node at every layer?
		x = np.array([1, 0, 1])


		self.assertTrue(evaluate_network(x, (W, V))[0] == 1)

	def test_bias(self):
		# check that the bias is properly encoded in the projective space representation
		# shift the signal up by +1 for the first node in the hidden layer. No bias for all the other nodes.
		W = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1]])

		V = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0,0,0,0], [0, 0, 0, 1]])


		x = np.array([1, 0, 1])

		# y = f(W.dot(x))
		# print(W.dot(x))
		# print(y)
		# print(V)
		# print(f(V.dot(y)))
		# print(f(V.dot(y))[0])


		self.assertTrue(np.all(f(W.dot(x)) == np.array([2, 0, 0, 1])))

		self.assertTrue(np.all(f(V.dot(f(W.dot(x)))) == np.array([2, 0, 0, 1])))

		self.assertTrue(evaluate_network(x, (W, V))[0] == 2)

if __name__ == "__main__":
	unittest.main()