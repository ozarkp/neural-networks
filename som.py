"""
Author: Patrick Ozark
som.py

Objectives:
	- Understand the Self-Organizing Map (SOM) by coding it from scratch
	- Be able to use NumPy to generate training data for SOM and other neural-net algorithms
	- Be able to use matplotlib to display results
"""

import numpy as np
import matplotlib.pyplot as plt

class SOM:

	def __init__(self,m,n):

		self.__m = m

		self.__n = n

		self.__u = (np.random.random((m,m,n)) / 10) + 0.45


	def plot(self,data):

		plt.scatter(data[:,0], data[:,1], s=.2), plt.gca().set_aspect('equal')

		for j in range(0,self.__m): #Search row and column indices

			for k in range(0,self.__m):

				#Plot neurons as red dots

				plt.plot(self.__u[j,k][0], self.__u[j,k][1], 'ro'), plt.gca().set_aspect('equal')

				#Plot blue lines between neurons and neighbors to create fencepost

				if j + 1 < self.__m:

					plt.plot((self.__u[j,k][0],self.__u[j+1,k][0]),(self.__u[j,k][1],self.__u[j+1,k][1]),'b-')

				if k + 1 < self.__m:

					plt.plot((self.__u[j,k][0],self.__u[j,k+1][0]),(self.__u[j,k][1],self.__u[j,k+1][1]),'b-')

		plt.show()


	def winner(self,e):

		values = []

		distances = []

		distanes = np.asarray(distances)

		indices = []

		for i in range(0,self.__m): #Find difference between e and the weights

			for j in range(0,self.__m):

				summation = np.sum((self.__u[i,j]-e)**2)

				distances=np.append(distances,summation)

				indices.append((i,j)) #Record the indices of the weights

		winner = np.argmin(distances) #Pick index of weight with smallest distance from e

		index = indices[winner] #Pick index tuple of weight with smallest distance from e

		return index







	def learn(self,data,T,a0,d0):

		for t in range(0,T):

			d = int(np.ceil(d0*(1-t/T))) #Compute current neighborhood radius d

			a = a0*(1-t/T) #Compute learning rate a

			e = data[np.random.random_integers(len(data)-1)] #Pick an index to use on data

			winner = self.winner(e)

			indices = self.neighbor(winner,d)

			for pair in indices:

				u = a * (e-self.__u[pair])

				self.__u[pair] += u


	def neighbor(self,p,d):

		indices = []

		for m in range(p[0]-d,p[0]+d+1): #Search row indices

			for n in range(p[1]-d,p[1]+d+1): #Search column indices

				if m < self.__m and m > -1 and n < self.__m and n > -1: #Ensure indices are within bounds of matrix size

					indices.append((m,n))

		return indices


#Create square-shaped training data

data = np.random.random((5000,2))

#Som object with square training data - initial conditions

som0 = SOM(8,2)
print("Evaluating initial conditions of square...")
som0.learn(data,0,.04,4)
plt.title("Self organizing map in square training data\n M=8   alpha0=0.04   d0=4   T=0")
som0.plot(data)

#Som object with square training data - final conditions

som1 = SOM(8,2)
print("Evaluating final conditions of square...")
som1.learn(data,4000,.04,4)
som0.learn(data,0,.04,4)
plt.title("Self organizing map in square training data\n M=8   alpha0=0.04   d0=4   T=4000")
som1.plot(data)

#Format training data into a ring

r = (data[:,0]-.5)**2 + (data[:,1]-.5)**2
data = data[np.logical_and(r<.2,r>.12)]


#Som object with ring training data - initial conditions

som2 = SOM(8,2)
print("Evaluating initial conditions of ring...")
som2.learn(data,0,.04,4)
plt.title("Self organizing map in a ring of training data\n M=8   alpha0=0.04   d0=4   T=0")
som2.plot(data)

#Som object with ring training data - final conditions

som3 = SOM(8,2)
print("Evaluating final conditions of ring...")
som3.learn(data,4000,.4,4)
plt.title("Self organizing map in a ring of training data\n M=8   alpha0=0.04   d0=4   T=4000")
som3.plot(data)
