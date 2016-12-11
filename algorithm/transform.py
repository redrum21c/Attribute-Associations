import networkx as nx
import algorithm.community as community
import numpy as np
import algorithm.kmeans as kmeans
import time
import sys
from collections import defaultdict
from scipy import stats
from scipy.stats import chisquare

# Make sure to turn off DEBUG when TIME_PROFILE is on
DEBUG = False
TIME_PROFILE = True

class GraphTrans:
	# configuration keys
	TIE_STRENGTH_CONFIG = 'tie_strength'
	PRUNING_CONFIG = 'pruning'
	MAX_ITERATION_CONFIG = 'max_iteration'
	SUPPORT_CONFIG = 'support'

	# configuration values
	TIE_STRENGTH_JACCARD = 'jaccard'
	TIE_STRENGTH_MINMAX = 'minmax'
	PRUNING_TRUE = 'true'
	PRUNING_FALSE = 'false'

	def __init__(self, G, model_config):
		# original graph
		self.G = G.copy()
		# alternative graph
		self.AG = nx.Graph()
		# number of attributes
		self.dim = len(G.node[G.nodes()[0]]['attrs'])
		# density of graph, defined as the ratio of the number of edges to the number of all possible edges
		self.density = G.number_of_edges() / float(G.number_of_nodes() * (G.number_of_nodes() - 1) * 0.5)
		# probabilities of having a value of 1 for each attribute
		self.p = self.get_attribute_probability()

		# initiate the graph AG with the list of nodes in the original graph
		# initially AG has only one cluster that contains all node in G
		self.AG.add_node(0)
		self.AG.node[0]['nodes'] = self.G.nodes()
		# number of ones for each attribute
		self.AG.node[0]['ones'] = [0] * self.dim
		for i in self.G.nodes():
			for j in range(self.dim):
				if self.G.node[i]['attrs'][j] == 1:
					self.AG.node[0]['ones'][j] += 1
		# which attribute already touched
		self.AG.node[0]['touched'] = [0] * self.dim
		# a list of clusters that should be considered a candidate for edge split
		self._block_list = []
		# current iteration
		self._iteration = 0
		# save the model configurations
		self._parse_configs(model_config)
		# set of statistically significant attribute associations
		self._significant_associations = []

		self.__setup_timer()

	def _parse_configs(self, model_config):
		# tie strength metric
		self.TIE_STRENGTH = model_config.get(self.TIE_STRENGTH_CONFIG, self.TIE_STRENGTH_MINMAX)
		if self.TIE_STRENGTH not in [self.TIE_STRENGTH_JACCARD, self.TIE_STRENGTH_MINMAX]:
			raise Exception('Not supported tie strength metric: %s' % self.TIE_STRENGTH)

		# pruning
		self.PRUNING = model_config.get(self.PRUNING_CONFIG, self.PRUNING_TRUE)
		if self.PRUNING not in [self.PRUNING_TRUE, self.PRUNING_FALSE]:
			raise Exception('Wrong configuration for pruning')

		# max iteration
		try:
			self.MAX_ITERATION = int(model_config.get(self.MAX_ITERATION_CONFIG, sys.maxint))
		except:
			raise Exception('Max iteration should be an integer value')

		# support
		try:
			self.SUPPORT = float(model_config.get(self.SUPPORT_CONFIG, 0.01))
		except:
			raise Exception('Support should be a float value')

	def __setup_timer(self):
		if TIME_PROFILE:
			print 'Time profiling enabled!'
			self.time_profile = dict()
			self.time_profile['timer'] = 0.
			self.time_profile['node_split'] = 0.
			self.time_profile['edge_split'] = 0.
			self.time_profile['node_split_cnt'] = 0
			self.time_profile['edge_split_cnt'] = 0

			self.time_profile['small_timer'] = 0.
			self.time_profile['edge_split:ts'] = 0.
			self.time_profile['edge_split:com'] = 0.

	def __start_timer(self):
		if TIME_PROFILE:
			self.time_profile['timer'] = time.time()

	def __add_to_timer(self, timer_name):
		if TIME_PROFILE:
			self.time_profile[timer_name] += (time.time() - self.time_profile['timer'])

	def __increment_counter(self, counter_name):
		if TIME_PROFILE:
			self.time_profile[counter_name] += 1

	def __start_small_timer(self):
		if TIME_PROFILE:
			self.time_profile['small_timer'] = time.time()

	def __add_to_small_timer(self, timer_name):
		if TIME_PROFILE:
			self.time_profile[timer_name] += (time.time() - self.time_profile['small_timer'])

	# --------------------------------------------
	# UTILITY FUNCTIONS
	# --------------------------------------------
	# computer p-values of subclusters and return maximum
	def get_max_subcluster_p_value(self, c):
		nodes = self.AG.node[c]['nodes']
		attrs = [self.G.node[u]['attrs'] for u in nodes]
		touched = np.array(self.AG.node[c]['touched'])
		k = 2

		# find an attribute which is least deviated from expectation
		cluster_size_threshold = self.SUPPORT * self.G.number_of_nodes()
		number_of_nodes = len(nodes)
		max_pval = 0.
		max_pval_attr = -1
		for i in range(self.dim):
			if self.AG.node[c]['touched'][i] != 1:
				# after splitting the cluster using an attribute,
				# at least one subcluster should satisfy the cluster size threshold
				if self.AG.node[c]['ones'][i] >= cluster_size_threshold or \
								(len(self.AG.node[c]['nodes']) - self.AG.node[c]['ones'][i]) >= cluster_size_threshold:
					pval = stats.binom_test(self.AG.node[c]['ones'][i], number_of_nodes, self.p[i],
					                        alternative='two-sided')
					if pval > max_pval:
						max_pval = pval
						max_pval_attr = i

		# in case of failing to find an attribute used for split,
		# then just return max p-value (1.0) so that node split will not be run on this cluster
		if max_pval_attr == -1:
			return 1.0

		# split the cluster based on the attribute with least deviation
		clusters = [[], []]
		for i in range(number_of_nodes):
			clusters[attrs[i][max_pval_attr]].append(nodes[i])

		max_p_value = 0.
		for i in range(k):
			nodes_i = clusters[i]
			sub_attrs = [self.G.node[j]['attrs'] for j in nodes_i]
			p_value = self.get_p_value_node(None, np.array([0] * self.dim), sub_attrs)
			if p_value > max_p_value:
				max_p_value = p_value

		return max_p_value

	# compute significance gain through node split
	def significance_gain_node_split(self, c):
		nodes = self.AG.node[c]['nodes']
		attrs = [self.G.node[u]['attrs'] for u in nodes]
		touched = np.array(self.AG.node[c]['touched'])
		k = 2

		# find an attribute which is least deviated from expectation
		number_of_nodes = len(nodes)
		max_pval = 0.
		max_pval_attr = -1
		for i in range(self.dim):
			if self.AG.node[c]['touched'][i] != 1:
				pval = stats.binom_test(self.AG.node[c]['ones'][i], number_of_nodes, self.p[i], alternative='two-sided')
				if pval > max_pval:
					max_pval = pval
					max_pval_attr = i

		# touched[max_pval_attr] = 1
		# split the cluster based on the attribute with least deviation
		clusters = [[], []]
		for i in range(number_of_nodes):
			clusters[attrs[i][max_pval_attr]].append(nodes[i])

		# compute significance gain: SG(C) = p(C) - (|C_1|/|C| * p(C_1) + |C_2|/|C| * p(C_2))
		sub_attrs = []
		significance_gain = self.get_p_value_node(c, touched)
		# print c, significance_gain,
		for i in range(k):
			nodes_i = clusters[i]
			sub_attrs = [self.G.node[j]['attrs'] for j in nodes_i]
			p_value = self.get_p_value_node(None, touched, sub_attrs)
			# print p_value,
			n_i = len(nodes_i)
			significance_gain -= ((float(n_i) / number_of_nodes) * p_value)

		# print '================='

		return significance_gain

	# get p-value of attribute similarity in the cluster c
	def get_p_value_node(self, c, touched, attrs=None):
		if attrs == None:
			nodes = self.AG.node[c]['nodes']
			attrs = [self.G.node[u]['attrs'] for u in nodes]

		# frequencies of 1s for each attribute
		zero_idx = np.where(touched == 0)[0]
		observed = [0] * len(zero_idx)
		for attr in attrs:
			for i in range(len(zero_idx)):
				if touched[zero_idx[i]] == 0:
					observed[i] += attr[zero_idx[i]]

		n = len(attrs)
		expected = [0] * len(zero_idx)
		for i in range(len(zero_idx)):
			expected[i] = n * self.p[zero_idx[i]]

		if len(observed) > 1:
			chisq, p_value = chisquare(observed, expected)
		else:
			p_value = stats.binom_test(observed[0], n, self.p[zero_idx[0]], alternative='two-sided')

		return p_value

	# get p-value of interaction weight (the number of edges) between two clusters
	def get_p_value_edge(self, c1, c2):
		n1 = len(self.AG.node[c1]['nodes'])
		n2 = len(self.AG.node[c2]['nodes'])
		observed = self.AG.edge[c1][c2]['weight']
		p_value = stats.binom_test(observed, n1 * n2, self.density, alternative='greater')

		return p_value

	# get the probabilities that each attribute has a value of 1
	def get_attribute_probability(self):
		p = [0] * self.dim

		for u in self.G.nodes():
			for i in range(self.dim):
				p[i] += self.G.node[u]['attrs'][i]

		for i in range(self.dim):
			p[i] = p[i] / float(self.G.number_of_nodes())

		return p

	# get the size of a cluster (the number of nodes in the cluster)
	def get_cluster_size(self, c):
		return len(self.AG.node[c]['nodes'])

	# get the number of edges between two clusters
	def count_edges(self, c1, c2):
		total = 0
		cluster1 = self.AG.node[c1]['nodes']
		cluster2 = self.AG.node[c2]['nodes']

		for i in cluster1:
			for j in cluster2:
				if self.G.has_edge(i, j):
					total += 1

		return total

	# get the first index available for new cluster in AG
	def findNodeNum(self):
		nodes = self.AG.nodes()
		for i in range(len(nodes)):
			if nodes[i] > i:
				return i

		return len(nodes)

	# check if there is an edge between a node(nodenum) and any nodes in the cluster
	def edgeExist(self, nodenum, c):
		for u in self.AG.node[c]['nodes']:
			if self.G.has_edge(nodenum, u):
				return True

		return False

	# get the number of edges between node u and nodes in the cluster c
	def get_number_of_edges(self, u, c):
		count = 0
		for v in self.AG.node[c]['nodes']:
			if self.G.has_edge(u, v):
				count += 1

		return count

	# get the average of all pairwise hamming distances in a cluster
	def hammingDist(self, c):
		# get a list of nodes in the cluster and corresponding attributes
		nodes = self.AG.node[c]['nodes']

		if len(nodes) == 1:
			return 0.0

		# collect all sets of attributes of the nodes
		attrs = []
		for i in nodes:
			attrs.append(self.G.node[i]['attrs'])
		attrs = np.asarray(attrs)

		n = len(attrs)
		totalDist = 0
		maxdist = 0

		for i in range(n):
			for j in range(i + 1, n):
				dist = 0
				# calculate the hamming distance between node i's attributes and node j's attributes
				for k in range(self.dim):
					if attrs[i][k] != attrs[j][k]:
						dist += 1

				totalDist += dist
				if dist > maxdist:
					maxdist = dist

		# return maxdist
		return totalDist / (n * (n - 1) * 0.5)

	# establish new connections between subclusters obtained from split and neighbor clusters
	def reconstruct(self, c, clusters, touched):  # c: old cluster to split / clusters: new subclusters after split
		# get neighbor clusters of the old cluster
		neighbors = self.AG.neighbors(c)
		# remove the old cluster from AG
		self.AG.remove_node(c)

		number_of_clusters = len(clusters)
		added = []

		for i in range(number_of_clusters):
			# get the first index available for the new cluster in AG
			new = self.findNodeNum()

			# add the new cluster to AG
			self.AG.add_node(new)
			self.AG.node[new]['nodes'] = clusters[i]
			# count the number of ones for each attribute for each cluster
			self.AG.node[new]['ones'] = [0] * self.dim
			for j in clusters[i]:
				for k in range(self.dim):
					if self.G.node[j]['attrs'][k] == 1:
						self.AG.node[new]['ones'][k] += 1
			if touched != None:
				self.AG.node[new]['touched'] = touched

			added.append(new)

			# make new connections between the i-th new subcluster and each of its neighbor clusters
			n1 = len(self.AG.node[new]['nodes'])
			for j in neighbors:
				# count the number of edges that exist between the new cluster and its j-th neighbor cluster
				number_of_edges = self.count_edges(new, j)
				if number_of_edges:
					# if there exist at least one edge between them, then create a new connection weighted by the number of edges between the clusters
					self.AG.add_edge(new, j, weight=number_of_edges)

		# make new connections between the new subclusters
		for i in range(number_of_clusters):
			c1 = added[i]
			n1 = len(self.AG.node[c1]['nodes'])
			for j in range(i+1, number_of_clusters):
				c2 = added[j]
				number_of_edges = self.count_edges(c1, c2)
				if number_of_edges:
					self.AG.add_edge(c1, c2, weight=number_of_edges)
	# --------------------------------------------

	# --------------------------------------------
	# NODE SPLIT (1ST PHASE)
	# --------------------------------------------
	def node_split(self, c):
		# get a list of nodes in the cluster and corresponding attributes
		nodes = self.AG.node[c]['nodes']
		attrs = []
		for i in nodes:
			attrs.append(self.G.node[i]['attrs'])

		# K-means clustering
		k = 2
		km = kmeans.Kmeans(attrs)
		idx = km.clustering(k)
		if len(idx) == 0:
			if DEBUG:
				print c, "contains identical attributes"
			return

		# new subclusters created after splitting cluster(c)
		clusters = []
		for i in range(k):
			clusters.append([])

		# assign each of the nodes in the cluster(c) to subclusters accorindg to the clustering result
		for i in range(len(idx)):
			clusters[idx[i]].append(nodes[i])

		# if there exists an empty cluster, then it should be removed
		for i in range(k):
			if len(clusters[i]) == 0:
				if DEBUG:
					print "cluster ", i, " empty!!"
				clusters.pop(i)

		# perform reconstruct AG so that the subclusters obtained from node split are connected to other clusters in AG
		self.reconstruct(c, clusters)

	# --------------------------------------------

	def node_split_binary(self, c):
		nodes = self.AG.node[c]['nodes']
		attrs = []
		for i in nodes:
			attrs.append(self.G.node[i]['attrs'])

		# find an attribute which is least deviated from expectation
		cluster_size_threshold = self.SUPPORT * self.G.number_of_nodes()
		number_of_nodes = len(nodes)
		max_pval = 0.
		max_pval_attr = -1
		for i in range(self.dim):
			if self.AG.node[c]['touched'][i] != 1:
				if self.AG.node[c]['ones'][i] >= cluster_size_threshold or \
								(len(self.AG.node[c]['nodes']) - self.AG.node[c]['ones'][i]) >= cluster_size_threshold:
					pval = stats.binom_test(self.AG.node[c]['ones'][i], number_of_nodes, self.p[i],
					                        alternative='two-sided')
					if pval > max_pval:
						max_pval = pval
						max_pval_attr = i

		clusters = [[], []]

		# split the cluster based on the attribute with least deviation
		for i in range(number_of_nodes):
			clusters[attrs[i][max_pval_attr]].append(nodes[i])

		# if there exists an empty cluster, then it should be removed
		for i in [1, 0]:
			if len(clusters[i]) == 0:
				if DEBUG:
					print "cluster ", i, " empty!!"
				clusters.pop(i)

		touched = self.AG.node[c]['touched'][:]
		touched[max_pval_attr] = 1

		self.reconstruct(c, clusters, touched)

	# --------------------------------------------
	# EDGE SPLIT (2ND PHASE)
	# --------------------------------------------
	def edge_split(self, c):
		# a graph with the nodes in the cluster c
		# node: a node in c
		# edge: exists if two ended nodes have edges to common clusters
		# edge is weighted by tie strength
		connGraph = nx.Graph()
		connGraph.add_nodes_from(self.AG.node[c]['nodes'])

		self.__start_small_timer()
		if TIME_PROFILE:
			self.time_profile['small_timer'] = time.time()

		if self.TIE_STRENGTH == self.TIE_STRENGTH_JACCARD:
			sets_of_neighbors = {}
			# for each node in c, find a set of clusters which are neighbors of c and have edges with the node
			for u in self.AG.node[c]['nodes']:
				neighbors = []
				for nb in self.AG.neighbors(c):
					if self.edgeExist(u, nb):
						if not (self.PRUNING and self.AG.edge[c][nb]['weight'] == 0):
							neighbors.append(nb)

				sets_of_neighbors[u] = neighbors

			for u in connGraph.nodes():
				for v in connGraph.nodes():
					if u < v:
						union = list(set(sets_of_neighbors[u]) | set(sets_of_neighbors[v]))
						intersection = list(set(sets_of_neighbors[u]) & set(sets_of_neighbors[v]))
						if len(union) == 0 or len(intersection) == 0:
							continue
						tie_strength = float(len(intersection)) / len(union)
						connGraph.add_edge(u, v, weight=tie_strength)

		elif self.TIE_STRENGTH == self.TIE_STRENGTH_MINMAX:
			number_of_edges = dict()
			for u in self.AG.node[c]['nodes']:
				number_of_edges[u] = defaultdict(int)
				for nb in self.AG.neighbors(c):
					if self.PRUNING and self.AG.edge[c][nb]['weight'] == 0:
						number_of_edges[u][nb] = 0
					else:
						number_of_edges[u][nb] = self.get_number_of_edges(u, nb)

			numerator = 0
			denominator = 0
			for u in connGraph.nodes():
				for v in connGraph.nodes():
					if u < v:
						for nb in self.AG.neighbors(c):
							numerator += min(number_of_edges[u][nb], number_of_edges[v][nb])
							denominator += max(number_of_edges[u][nb], number_of_edges[v][nb])

						if numerator == 0 or denominator == 0:
							continue
						tie_strength = float(numerator) / denominator
						connGraph.add_edge(u, v, weight=tie_strength)
		self.__add_to_small_timer('edge_split:ts')

		# find nodes without neighbors and remove them from connGraph
		disconnectedNode = []
		for u in connGraph.nodes():
			if len(connGraph.neighbors(u)) == 0:
				disconnectedNode.append(u)
				connGraph.remove_node(u)

		self.__start_small_timer()
		if connGraph.number_of_nodes() > 0:
			# perform graph partitioning based on the weights(tie strength)
			# return a dict with pairs of (key:node_id, value: partition assignment)
			partitions = community.best_partition(connGraph)
			numberOfPartitions = max(partitions.values()) + 1
		else:
			partitions = {}
			numberOfPartitions = 0
		self.__add_to_small_timer('edge_split:com')

		# all disconnected nodes will belong to the same subcluster after edge split
		if len(disconnectedNode) > 0:
			for u in disconnectedNode:
				# disconnected node 'u' is assigned to the partition 'numberOfPartitions', '...'n
				partitions[u] = numberOfPartitions
			numberOfPartitions += 1

		if DEBUG:
			print "the number of partitions is", numberOfPartitions, '...',

		if numberOfPartitions == 1:
			self._block_list.append(c)
			if DEBUG:
				print c, 'is blocked due to no partitioning...',
			return

		# print numberOfPartitions
		nodes = self.AG.node[c]['nodes']
		# create subclusters obtained from edge split
		clusters = [[] for i in range(numberOfPartitions)]
		for i in partitions.keys():
			clusters[partitions[i]].append(i)

		# there must be at least one subcluster that satisfies support (cluster size threshold)
		flag = True
		cluster_size_threshold = self.SUPPORT * self.G.number_of_nodes()
		for subcluster in clusters:
			if len(subcluster) >= cluster_size_threshold:
				flag = False
				break

		# if there are no subclusters satisfying support, then edge split is cancelled and the cluster is blocked
		if flag:
			self._block_list.append(c)
			if DEBUG:
				print c, 'is blocked due to too small subclusters...',
			return

		touched = self.AG.node[c]['touched'][:]
		# make new connections between the subcluesters and their neighbor clusters in AG
		self.reconstruct(c, clusters, touched)

	# --------------------------------------------

	def reset(self):
		self.__init__(self.G)

	def run(self, max_iter=None, support=None):
		if max_iter is None:
			max_iter = self.MAX_ITERATION
		if support is None:
			support = self.SUPPORT

		iter_start = self._iteration
		self._iteration += max_iter

		cluster_size_threshold = support * self.G.number_of_nodes()
		strength_threshold = cluster_size_threshold * cluster_size_threshold * self.density

		for i in xrange(iter_start, self._iteration):
			if DEBUG:
				print "======== Iteration", i, '========'
			self.__start_timer()
			# NODE SPLIT
			min_p_value = 1
			cluster_to_node_split = -1
			if DEBUG:
				# print 'calculating the max subcluster p-value of the cluster', c, '...',
				print 'finding a cluster for node split...',
				start_time = time.time()

			for c in self.AG.nodes():
				# check statistical significance of the attribute similarity in the cluster c and the number of nodes in c
				# perform node split only for a cluster that is not significant and satisfies the support
				if len(self.AG.node[c]['nodes']) > cluster_size_threshold:
					# check if there is at least one attribute which has not been touched yet
					cnt = 0
					for j in range(self.dim):
						if self.AG.node[c]['touched'][j] == 0:
							cnt += 1
					if cnt == 0:
						continue

					# find a cluster that will produce after split
					max_subcluster_p_value = self.get_max_subcluster_p_value(c)

					if max_subcluster_p_value < min_p_value:
						min_p_value = max_subcluster_p_value
						cluster_to_node_split = c

			if DEBUG:
				elapsed = time.time() - start_time
				print 'done (%.3f)' % elapsed

			# if there is no cluster with gain, then just skip node split
			if cluster_to_node_split != -1:
				if DEBUG:
					print 'node splitting on cluster', cluster_to_node_split, '...',
					start_time = time.time()

				# if the cluster to split is in the block list, then remove it from the list
				if cluster_to_node_split in self._block_list:
					self._block_list.remove(cluster_to_node_split)
					if DEBUG:
						print cluster_to_node_split, 'is removed from block list...',

				self.node_split_binary(cluster_to_node_split)
				self.__increment_counter('node_split_cnt')

				if DEBUG:
					elapsed = time.time() - start_time
					print 'done (%.3f)' % elapsed

			self.__add_to_timer('node_split')

			self.__start_timer()
			# EDGE SPLIT
			max_p_value = 0
			cluster_to_edge_split = -1

			if DEBUG:
				print 'finding a cluster for edge split...',
				start_time = time.time()

			for c in self.AG.nodes():
				if c not in self._block_list and len(self.AG.node[c]['nodes']) > cluster_size_threshold and len(
						self.AG.edge[c]) > 1:
					# check if there is at least one interaction with strength higher than threshold
					flag = True
					for nb in self.AG.edge[c]:
						if self.AG.edge[c][nb]['weight'] >= strength_threshold:
							flag = False
							break
						elif self.PRUNING:
							self.AG.edge[c][nb]['weight'] = 0
					if flag:
						# if all interactions have lower strength than threshold then no edge split
						continue

					sum_p_value = 0
					for nb in self.AG.edge[c]:
						sum_p_value += self.get_p_value_edge(c, nb)
					avg_p_value = sum_p_value / len(self.AG.edge[c])
					if avg_p_value > max_p_value:
						max_p_value = avg_p_value
						cluster_to_edge_split = c

			if DEBUG:
				elapsed = time.time() - start_time
				print 'done (%.3f)' % elapsed

			if cluster_to_edge_split != -1 and max_p_value > 0:
				if DEBUG:
					print 'edge splitting on cluster', cluster_to_edge_split, '...',
					start_time = time.time()

				self.edge_split(cluster_to_edge_split)
				self.__increment_counter('edge_split_cnt')

				if DEBUG:
					elapsed = time.time() - start_time
					print 'done (%.3f)' % elapsed

			self.__add_to_timer('edge_split')

			if cluster_to_node_split == -1 and cluster_to_edge_split == -1:
				return self.run_done()

		return self.run_done()

	def run_done(self):
		print '############# No more clusters to split #############'
		print '############# GraphTran algorithm DONE  #############'
		if TIME_PROFILE:
			print 'Time elapsed (node split,edge split): ({:.3f},{:.3f})'.format(self.time_profile['node_split'],
			                                                                     self.time_profile['edge_split'])
			return self.time_profile

	# --------------------------------------------
	# RESULT ANALYSIS FUNCTIONS
	# --------------------------------------------
	# show the current status including statistics
	def showStatus(self):
		# print "=== NODES ==="
		sum_of_size = 0
		sum_of_dist = 0
		for n in self.AG.nodes():
			# print "Node", n, ": ", len(self.AG.node[n]['nodes']), "(maximum hamming distance=", self.hammingDist(n), ")"
			# print "Node", n, ": ", len(self.AG.node[n]['nodes']), self.calBinomProb(n), self.calMaxMinBinomProb(n)
			sum_of_size += len(self.AG.node[n]['nodes'])
			sum_of_dist += self.hammingDist(n)

		# print "=== EDGES ==="
		for n in self.AG.nodes():
			neighbors = self.AG.neighbors(n)
			edges = {}
			for nb in neighbors:
				# edges[nb] = str(self.AG.edge[n][nb]['weight']) + '/' + str(int(len(self.AG.node[n]['nodes']) * len(self.AG.node[nb]['nodes']) * self.edgeProb))
				edges[nb] = round(self.AG.edge[n][nb]['weight'] / (
				len(self.AG.node[n]['nodes']) * len(self.AG.node[nb]['nodes']) * self.density), 2)

			avg_strength_conn = 0
			if len(neighbors) != 0:
				avg_strength_conn = sum(edges.values()) / len(neighbors)
			# print "Node", n, ": ", avg_strength_conn, edges
			# print "Node", n, ": ", avg_strength_conn, edges.keys()

		print "=== SUMMARY ==="
		sum_of_strength = 0
		num_strong = 0
		num_weak = 0
		for [u, v] in self.AG.edges():
			n1 = len(self.AG.node[u]['nodes'])
			n2 = len(self.AG.node[v]['nodes'])
			expected = n1 * n2 * self.density
			strength = round(self.AG.edge[u][v]['weight'] / expected, 2)

			pval = stats.binom_test(self.AG.edge[u][v]['weight'], n1 * n2, self.density, alternative='greater')
			if pval <= 0.001:
				num_strong += 1
			elif pval >= 0.999:
				num_weak += 1

			sum_of_strength += strength

		# number of nodes in alternative network
		print "NUMNODES:", self.AG.number_of_nodes()
		# average size of clusters
		print "AVGSIZE: ", sum_of_size / self.AG.number_of_nodes()
		# average similarity of clusters
		print "AVGDIST: ", sum_of_dist / self.AG.number_of_nodes()
		# number of edges in alternative network
		print "NUMEDGES:", self.AG.number_of_edges()
		# average strength of connectivities
		print "AVGSTRG: ", sum_of_strength / self.AG.number_of_edges()
		# number of strong connections (strength > 3)
		print "STRCONN: ", num_strong
		# number of weak connections (strength < 0.5)
		print "WEAKCONN:", num_weak

	# save AG (association graph) to a file
	def save_model(self, filename):
		print 'saving the association graph to the file:', filename
		with open(filename, 'w') as f:
			for (u, v) in self.AG.edges():
				# for each edge in AG, all information should be tab-separated
				# (cluster1_id, cluster2_id, strength, cluster1_nodes, cluster1_ones, cluster1_touched, cluster2...)
				f.write(str(u) + '\t' + str(v) + '\t' + str(self.AG.edge[u][v]['weight']) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[u]['nodes']]) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[u]['ones']]) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[u]['touched']]) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[v]['nodes']]) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[v]['ones']]) + '\t')
				f.write(','.join([str(x) for x in self.AG.node[v]['touched']]) + '\n')

	# load AG (association graph) from a file
	def load_model(self, filename):
		print 'loading the association graph from the file:', filename
		self.AG = nx.Graph()
		with open(filename) as f:
			for row in f.readlines():
				tokens = row.split('\t')
				u = int(tokens[0])
				v = int(tokens[1])
				strength = int(tokens[2])
				self.AG.add_edge(u, v, weight=strength)
				self.AG.node[u]['nodes'] = [int(x) for x in tokens[3].split(',')]
				self.AG.node[u]['ones'] = [int(x) for x in tokens[4].split(',')]
				self.AG.node[u]['touched'] = [int(x) for x in tokens[5].split(',')]
				self.AG.node[v]['nodes'] = [int(x) for x in tokens[6].split(',')]
				self.AG.node[v]['ones'] = [int(x) for x in tokens[7].split(',')]
				self.AG.node[v]['touched'] = [int(x) for x in tokens[8].split(',')]


if __name__ == '__main__':
	from data.yelp import yelp_academic_dataset_handler

	handler = yelp_academic_dataset_handler.YelpAcademicDatasetHandler()
	G = handler.load('../data/yelp/yelp_TOP_1_r20.graphml')

	gt = GraphTrans(G)
	gt.run()
