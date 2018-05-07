"""
Author: Patrick Ozark
backprop.py
Objectives:
	- Apply the delta rule in back propogation to implement the XOR logic gate
	- Understand the use of hidden layers in supervised learning, feedforward neural networks
"""

import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(y):
	return y * (1-y)

class Backprop:

	def __init__(self, nx, nh, ny):

		# Start with random weights centered at zero
		self.wxh = np.random.randn(nx, nh)
		self.why = np.random.randn(nh, ny)

		# Same for biases
		self.bh = np.random.randn(nh)
		self.by = np.random.randn(ny)


	def train(self, inp, tgt, niter, alpha):

		for i in range(niter):

			for p in range(len(inp)):

				# Grab vector for current input pattern
				x = inp[p]

				# Compute hidden layer as sigmoid of input * weights, plus bias
				h = sigmoid(np.dot(x, self.wxh) + self.bh)

				# Compute output layer as sigmoid of hidden * weights, plus bias
				y = sigmoid(np.dot(h, self.why) + self.by)

				# Compute error on output by comparing against target
				ey = tgt[p] - y

				# Compute "little delta" on output
				dy = ey * dsigmoid(y)

				# Compute error on hidden using backprop
				eh = np.dot(dy, self.why.T)

				# Compute "little delta" on hidden
				dh = eh * dsigmoid(h)

				# Adjust weights using Delta Rule
				self.why += alpha * np.outer(h, dy)
				self.by += alpha * dy
				self.wxh += alpha * np.outer(x, dh)
				self.bh += alpha * dh

				# Every hundred iterations, report output error for current iteration, pattern
				if i%100 == 0:
					print(i,x,np.abs(ey))

	def test(self, inp):

		for p in range(len(inp)):

			# Grab vector for current input pattern
			x = inp[p]

			# Compute hidden layer as sigmoid of input * weights, plus bias
			h = sigmoid(np.dot(x, self.wxh) + self.bh)

			# Compute output layer as sigmoid of hidden * weights, plus bias
			y = sigmoid(np.dot(h, self.why) + self.by)

			print(x, y)

def main():
	inp = np.array([ [0,0], [0,1], [1,0], [1,1]])
	tgt = np.array([ [0],   [1],   [1],   [0]   ])

	net = Backprop(2, 3, 1)

	net.train(inp, tgt, 5000, 0.5)

	print("\nTest on learning XOR: ")
	net.test(inp)

main()
