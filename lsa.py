"""
Author: Patrick Ozark
lsa.py
Objectives:
	- Apply latent semantic analysis (LSA) to the real-world oproblem f document retrieval using a pilot document with high text similarity
"""

import numpy as np


def show_lsa(docs):
	"""
	Show the latent semantic analysis of inputted documents
	"""
	docNum = len(docs)
	d = {}

	#loop for the number of docs, creating a dictionary entry for each one
	for i in range(0, docNum):
		d['d{}'.format(i+1)] = np.zeros(len(terms),dtype=int)
		d['d{}'.format(i+1)] = d['d{}'.format(i+1)]

		#loop through terms, if term is in a doc, add 1 at index in d correspoding to index in terms
		for term in docs[i].split(" "):

			if term in terms:
				term_index = terms.index(term)
				d['d{}'.format(i+1)][term_index] +=1


	sorted_keys = sorted(d.keys())


	print("Step 1: Construct term-document matrix A and query matrix q:\n")

	q = build_query(terms, ["gold","silver","truck"])


	fmt = '{!s:>17}{!s:>8}{!s:>8}{!s:>16}'
	print(fmt.format("d1","d2","d3","q"),"\n")


	fmt1 = '{!s:<15}{!s:<8}{!s:<8}{!s:<17}{!s}' #Use !s to force format parameter to string

	for (term,d1,d2,d3,query) in zip(terms,d['d1'],d['d2'],d['d3'],q):
		print(fmt1.format(term,d1,d2,d3,query))





	print("\n\nStep 2: Decompose matrix A into U, S, V^T:\n")

	A = np.asarray([d['d1'],
					d['d2'],
					d['d3']]).transpose()

	U,S,Vt = np.linalg.svd(A, full_matrices=False)
	S = np.diag(S)
	V = Vt.transpose()
	print("U:\n",U,"\n")
	print("S:\n",S,"\n")
	print("V^T:\n",Vt,"\n")



	print("\n\nStep 3: Implement a Rank 2 Approximation by keeping\
the first two columns of U and V and the first two \
columns and rows of S:\n")

	U_k = U[:,:2] #Select first 2 columns of U
	S_k = S[:2,:2] #Select first 2 columns and rows of S
	V_k = V[:,:2] #Select first 2 columns of V, then transpose to get V_k^T
	V_kt = V_k.transpose()

	print("U_k:\n",U_k,"\n")
	print("S_k:\n",S_k,"\n")
	print("V_k^T:\n",V_kt,"\n")



	print("\nStep 4: Find the new document vector coordinates in this reduced 2-dimensional space. \
Rows of V holds eigenvector values. These are the coordinates of individual document \
vectors, hence:\n")

	print("V:\n",V_k)


	print("\n\nStep 5: Find the new query vector coordinates in the \
reduced 2-dimensional space:\n")

	Sk_inv = np.linalg.inv(S_k)
	print("Sk^-1:\n",Sk_inv,"\n")

	q = np.dot(q,np.dot(U_k,Sk_inv))
	print("q:\n",q)



	print("\n\nStep 6: Rank documents in decreasing order \
of query-document cosine similarities:\n")

	print("sim(q, d1) = ", "%.6f" % sim(q, V_k[0]))
	print("sim(q, d2) = ", "%.6f" % sim(q, V_k[1]))
	print("sim(q, d3) = ", "%.6f" % sim(q, V_k[2]))


def build_query(terms, query_list):
	""""
	Build a query against document terms and query words
	"""
	q = np.zeros(len(terms),dtype=int)

	for term in query_list:
		if term in terms:
			term_index = terms.index(term)
			q[term_index] += 1
	return q

def mag(vec):
	"""V
	Vector magnitude
	"""
	return np.sqrt(np.sum(vec**2))

def cosine(vec1,vec2):
	"""
	Cosine similarity
	"""
	return np.dot(vec1,vec2)/(mag(vec1) * mag(vec2))



if __name__ == '__main__':

	docs = ['shipment of gold damaged in a fire',
			'delivery of silver arrived in a silver truck',
			'shipment of gold arrived in a truck']

	terms = set()
	for doc in docs:
		doc = doc.split(" ")

		for term in doc:
			if term not in terms:
				terms.add(term)

	terms = sorted(terms)

	np.set_printoptions(precision=4)

	show_lsa(docs)
