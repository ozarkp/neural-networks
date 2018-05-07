"""
Author: Patrick Ozark
vsa.py
Objectives:
	- Understand vector symbolic architectures (VSA) by implementing the "Dollar of Mexico" example from Kanerva (2008)-- https://www.aaai.org/ocs/index.php/FSS/FSS10/paper/viewFile/2243/2691
	- Implement permutation to implement sequencing
	- Apply Ross Gayler's Multiply/Add/Permute flavor of VSA using NumPy
"""
import numpy as np

class VSA:

	def __init__(self,n):
		self.n = n
		self.d = {}
		self.indices = np.random.permutation(n)


	def randvec(self,name):
		"""
		Return random vector of -1's and 1's of size n
		"""
		vec = (2 * np.random.random_integers(0, 1, size = self.n)) - 1
		self.d[name] = vec
		return vec


	def mag(self,vec):
		"""
		Vector magnitude
		"""
		return np.sqrt(np.sum(vec**2))


	def cosine(self,vec1,vec2):
		"""
		Cosine similarity
		"""
		return np.dot(vec1,vec2)/(self.mag(vec1) * self.mag(vec2))


	def winner(self,vec):
		"""
		Find vector name with highest similarity to input vector
		"""
		cosines = []
		keys = []

		for key in self.d.keys():

			keys.append(key)

			cosines.append(self.cosine(self.d[key], vec))

		i = cosines.index(max(cosines))

		return keys[i]


	def permute(self,vec):
		"""
		Permute input vector
		"""
		permuted_vec = []
		permuted_vec = np.asarray(permuted_vec, dtype='int')

		for i in self.indices:

			permuted_vec = np.append(permuted_vec,vec[i])

		return permuted_vec


	def perminv(self,vec):
		"""
		Permute inverse of input permuted vector
		"""
		return vec[np.argsort(self.indices)]


	def seqencode(self,symbols):
		"""
		Encode a sequence of symbols
		"""
		encoded = np.zeros(self.n, dtype = 'int')

		# Loop through reverse symbols
		for symbol in symbols[::-1]:

			encoded += self.d[symbol]

			encoded = self.permute(encoded)

		return encoded


	def seqdecode(self, encoded):
		"""
		Decode an encoded sequence, returning the symbols for
		the vectors that composed the encoded sequence
		"""
		symbols = []
		symbols = np.asarray(symbols)

		# Establish baseline cosine to be compared with other cosines when decoding
		ref = self.perminv(encoded)
		ref_winner = self.winner(ref)
		ref_cosine = self.cosine(self.d[ref_winner], ref)

		while True:

			encoded = self.perminv(encoded)
			winner = self.winner(encoded)
			winner_cosine = self.cosine(self.d[winner], encoded)

			# Encoded vector will have a similarity close to the similarity of the first vector
			if abs(winner_cosine - ref_cosine) < 0.1:

				symbols = np.append(symbols,winner)

			else:

				return symbols
