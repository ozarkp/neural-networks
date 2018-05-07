"""
Author: Patrick Ozark
sdm.py
Objectives:
	- Understand Kanerva's Sparse Distributed Memory by coding it from scratch
	- Reproduce the image-restoration results in Denning (1989)-- http://denninginstitute.com/pjd/PUBS/amsci-sdm.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class SDM:

	def __init__(self,p,n):

		self.p = p

		self.n = n

		self.d = 0.451 * n

		self.addresses = np.random.randint(2, size=(p,n))

		self.data = np.zeros((p,n),dtype = int)




	def enter(self, vector):

		#Loop through addresses and calculate Hamming distance
		for i in range(self.p):

			r = np.sum(self.addresses[i] != vector)

			#Add one to corresponding position in data where vector == 1
			#Subtract in data where vector == 0
			if r <= self.d:

				for j in range(self.n):

					if vector[j] == 1:

						self.data[i,j] += 1

					else:

						self.data[i,j] -= 1





	def lookup(self, vector):

		#Create new data vector
		newData = np.zeros(self.n, dtype = int)

		#Compute Hamming distance and add
		for i in range(self.p):

			r = np.sum(self.addresses[i] != vector)

			if r <= self.d:

				newData += self.data[i]

		#Set all values to 1 in new data vector if value > 0
		#and to 0 if value < 0

		return (newData > 0).astype('int')


if __name__ == '__main__':

	#Create 256 bit test parttern of 0s and 1s
	testpat = np.random.random_integers(0, 1 , 256)

	def plot(pattern, columns):

		i = 0

		while i < len(pattern):

			for columnIndex in range(columns):

				#Print elements of pattern, replace 0 with " " and 1 with "*"
				print(str(pattern[i]).replace("0"," ").replace("1", "*"), end = " ")

				i += 1

			print("")


	def ring():

		return np.asarray([0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,
						   0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,
						   0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,
						   0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,
						   0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
						   0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,
						   0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,
						   0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,
						   0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,
						   0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,])

	def noisy_copy(vector, probability):

		#Flip bits in the vector according to the input probability
		for i in range(len(vector)):

			rand = random.random()

			if probability > rand:

				if vector[i] == 1:

					vector[i] = 0

				else:

					vector[i] = 1

		return vector


	print("Part 4: Recover pattern afer 25% noise added\n")

	r = ring()
	sdm = SDM(2000,256)
	sdm.enter(r)

	noisy = noisy_copy(r, 0.25)
	plot(noisy, 16)

	print("\n")

	plot(sdm.lookup(noisy),16)

	print("\n")

	print("Part 5: Learn with the following five noisy examples: \n")
	r1 = ring()
	sdm1 = SDM(2000, 256)
	noisy1 = noisy_copy(r1, 0.10)
	plot(noisy1, 16)
	print("\n")

	r2 = ring()
	noisy2 = noisy_copy(r2, 0.10)
	plot(noisy2, 16)
	print("\n")

	r3 = ring()
	noisy3 = noisy_copy(r3, 0.10)
	plot(noisy3, 16)
	print("\n")

	r4 = ring()
	noisy4 = noisy_copy(r4, 0.10)
	plot(noisy4, 16)
	print("\n")

	r5 = ring()
	noisy5 = noisy_copy(r5, 0.10)
	plot(noisy5, 16)
	print("\n")

	sdm1.enter(noisy1)
	sdm1.enter(noisy2)
	sdm1.enter(noisy3)
	sdm1.enter(noisy4)
	sdm1.enter(noisy5)

	print("Test with the following probe: \n")
	r6 = ring()
	probe = noisy_copy(r6, 0.10)
	plot(probe, 16)

	print("\n")

	print("Result: \n")
	plot(sdm1.lookup(probe),16)
