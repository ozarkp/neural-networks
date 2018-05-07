"""
Author: Patrick Ozark
hopfield.py
Objectives:
	- Understand the Hopfield Network by coding it from scratch
	- Understand the limitations of the network through empirical testing
"""

import numpy as np
import random

class Hopfield:

	def __init__(self,n):

		self.T = np.zeros((n,n))


	def learn(self,data):

		for a in data: #Loop through training data and update weights based on outer product

			self.T += np.outer(2*a - 1,2*a - 1)

		np.fill_diagonal(self.T,val=0.0)



	def test(self,vector,iterations = 5):

		for iteration in range(iterations):

			vector = (np.dot(self.T,vector) >= 0).astype('int')

		return vector



if __name__ == '__main__':

	data0 = np.random.randint(2,size=(5,30))

	def show_confusion(array1,array2):

		for i in range(0,len(array1)):

			similarities = [] #List to store similarities (This assists with formatted output)

			tracker = i #Track index i to control how many times to print out the similarity
						#Only want to print the values in the similarities list i+1 times

			for j in range(0,len(array1)):

				similarity = np.sum(array1[i]*array2[j]) / ((np.sqrt(np.sum(array1[i]**2)) * np.sqrt(np.sum(array2[j]**2))))

				similarities.append(similarity)

			index = 0 #Index of the similarities list to print out

			while tracker > -1:

				print("%.2f" % similarities[index],end= " ")

				index += 1

				tracker -= 1

			print("")



	def noisy_copy(array1, p):

		noisy = np.copy(array1)

		for i in range(0,len(noisy)): #Loop through every element of the data and flip
									  #the value of the bit with probability p

			for j in range(0,len(noisy[i])):

				randInt = random.random()

				if p > randInt:

					if noisy[i,j] == 1:

						noisy[i,j] = 0

					else:

						noisy[i,j] = 1

		return noisy



	print("\nPart 2: Vector-cosine confusion matrix of an array with itself ------------\n")

	show_confusion(data0,data0) #Show cosine similarity of array with itself

	print("\n")

	noisy0 = noisy_copy(data0,.25) #Add noise with 25% probability to the array
							     #and show confusion
	print("Part 3: Confusion matrix with 25 percent noise ------------\n")

	show_confusion(noisy0,data0)

	print("\n")

	hopfield0 = Hopfield(30)

	hopfield0.learn(data0)

	print("Part 4: Recovering small patterns with a Hopfield net ------------\n")
	print("Recover pattern, no noise:")
	randIndex = np.random.random_integers(0,len(data0)-1) #Pick random vector from training data, inclusive
	print("Input: ", data0[randIndex])
	output = hopfield0.test(data0[randIndex]) #Recover original vector from input vector
	print("Output: ",output)
	print("Vector cosine: ",np.sum(data0[randIndex]*output) / ((np.sqrt(np.sum(data0[randIndex]**2)) * np.sqrt(np.sum(output**2)))))
	print("\n")

	print("Recover pattern, 25% noise:")
	print("Input: ",noisy0[randIndex])
	output = hopfield0.test(noisy0[randIndex]) #Recover original vector from noisy vector
	print("Output: ",output)
	print("Original: ",data0[randIndex])
	print("Vector cosine: ",np.sum(data0[randIndex]*output) / ((np.sqrt(np.sum(data0[randIndex]**2)) * np.sqrt(np.sum(output**2)))))
	print("\n")

	data1 = np.random.randint(2,size=(10,1000)) #Increase capacity using 10x1000 training data set

	hopfield1 = Hopfield(1000)

	hopfield1.learn(data1)

	noisy1 = noisy_copy(data1,.25)

	print("Part 5: Recovering big patterns ------------\n")
	print("Confusion matrix for 1000-element vectors with 25% noise:\n")
	show_confusion(noisy1,data1)
	print("\n")
	print("Recovering patterns with 25% noise:\n")

	for i in range(len(noisy1)): #Loop through every vector of the noisy matrix
		output = hopfield1.test(noisy1[i]) #Recover original vector form noisy vector
		print("Vector cosine on pattern ",i,"= ", np.sum(data1[i]*output) / ((np.sqrt(np.sum(data1[i]**2)) * np.sqrt(np.sum(output**2)))))
