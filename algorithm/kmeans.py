'''
K-means clustering(k=2) based on hamming distance between two attribute vectors.

@author: Jihwan Lee (jihwan@purdue.edu)
'''

import sys
import numpy as np
import random

def getDescription(attrs):
	dimensions = len(attrs[0])
	number_of_attrs = len(attrs)

	stats = {}
	values = {}
	distributions = {}

	for i in range(dimensions):
		p = []
		for attr in attrs:
			p.append(attr[i])

		p = set(p)
		values[i] = list(p)

	for i in range(dimensions):
		stats[i] = len(values[i])

	for i in range(dimensions):
		dist = {}
		for key in values[i]:
			dist[key] = 0

		for attr in attrs:
			dist[attr[i]] += 1

		distributions[i] = dist

	return stats, values, distributions

class Kmeans():
	def __init__(self, data):
		# input data (a set of attribute vectors(nodes))
		self.data = data
		# centroids of k clusters
		self.centroids = []
		# cluster assignments for each attribute vector
		self.cluster_membership = []
		# number of attribute vectors
		self.number_of_attrs = len(data)
		[self.stats, self.values, self.distributions] = getDescription(data)

		self.distMatrix = np.zeros(shape=(self.number_of_attrs, self.number_of_attrs))
		for i in range(self.number_of_attrs):
			for j in range(i, self.number_of_attrs):
				dist = self.matchingdist(data[i],data[j])
				self.distMatrix[i, j] = dist
				self.distMatrix[j, i] = dist

	def getSimilarity(self, x, y):
		d = len(x)
		sim = 0

		for i in range(d):
			if x[i] == y[i]:
				Q = []
				for q in self.values[i]:
					if (self.distributions[i][q]/self.number_of_attrs) <= (self.distributions[i][x[i]]/self.number_of_attrs):
						Q.append(q)

				sim_k = 0
				for q in Q:
					sim_k += pow((self.distributions[i][q]/self.number_of_attrs),2)
				sim += (1-sim_k) * (1 / float(d))

		return sim

	def matchingdist(self, c, d):
		dist = 0
		dim = len(c)
		for i in range(dim):
			if c[i] != d[i]:
				n_c = self.distributions[i][c[i]]
				n_d = self.distributions[i][d[i]]
				dist += float(n_c + n_d) / (n_c * n_d)

		return dist

	def clustering(self, k):
		self.centroids = random.sample(range(len(self.data)), k)

		prev_centroids = [0] * k
		self.cluster_membership = [0] * len(self.data)

		while self.centroids != prev_centroids:
			prev_centroids = list(self.centroids)

			######## Assign cluster #########################
			for d in range(len(self.data)):
				min_dist = sys.maxint

				for i in range(k):
					#dist = self.matchingdist(self.data[self.centroids[i]], self.data[d])
					dist = self.distMatrix[self.centroids[i], d]

					if dist < min_dist:
						min_dist = dist
						self.cluster_membership[d] = i
			#################################################

			######## recalculate centroids ##################
			for i in range(k):
				new_centroid = []

				min_dist = sys.maxint
				new_index = 0
				for j in range(len(self.cluster_membership)):
					if i == self.cluster_membership[j]:
						total_dist = 0
						for p in range(len(self.cluster_membership)):
							if i == self.cluster_membership[p]:
								#total_dist += self.matchingdist(self.data[j], self.data[p])
								total_dist += self.distMatrix[j,p]

						if total_dist < min_dist:
							min_dist = total_dist
							new_index = j


				self.centroids[i] = new_index
			#################################################

		return self.cluster_membership
