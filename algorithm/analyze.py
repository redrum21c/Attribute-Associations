import numpy as np
from collections import defaultdict
from collections import Counter
import networkx as nx
import scipy.misc
from scipy import stats

ALPHA_FOR_SIGNIFICANT_PATTERNS = 0.01
ALPHA_FOR_STAR_RESOLUTIONS = 0.01

# write the result to a file
def save_result(model, k, filename, show_stars=True, remove_homophily=False, show_graph_patterns=True, \
                config_filename=None, data_configs=None, model_configs=None):
	#tops = _get_significant_patterns(model, remove_homophily)[:k]
	tops = _get_significant_patterns(model, remove_homophily)
	top_frequents = _get_frequent_patterns(model, remove_homophily)[:k]

	attrs = [tuple(model.G.node[node_id]['attrs']) for node_id in model.G.nodes()]
	attr_counts = Counter(attrs)
	d = nx.density(model.G)

	cluster_size_threshold = model.SUPPORT * model.G.number_of_nodes()

	with open(filename, 'w') as f:
		if config_filename:
			f.write('{0:35s}: {1:s}\n\n'.format('Config filename', config_filename))
		if data_configs:
			for key, val in data_configs.items():
				f.write('data.{0:30s}: {1:s}\n'.format(key,str(val)))
		if model_configs:
			for key, val in model_configs.items():
				f.write('model.{0:29s}: {1:s}\n'.format(key,str(val)))
		if data_configs or model_configs:
			f.write('\n')

		column1 = ['Original graph'] + nx.info(model.G).split('\n')[2:] + ['Graph density: %f' % nx.density(model.G)]
		column2 = ['Association graph'] + nx.info(model.AG).split('\n')[2:] + ['Graph density: %f' % nx.density(model.AG)]
		for c1, c2, in zip(column1, column2):
			f.write('{0:35s} {1:35s}\n'.format(c1,c2))
		f.write('Attr full names: %s' % (str(model.G.name)))
		f.write('\n')

		# Output TOP SIGNIFICANT ASSOCIATIONS
		f.write('{0:4s}({1:3s}) {2:4s}({3:3s}) {4:3s} {5:3s}\n'.format('C1', 'siz', 'C2', 'siz', 'pvl', 'str'))
		for top in tops:
			f.write('{0:4d}({1:3d}) {2:4d}({3:3d}) {4:0.3f} {5:3d}'.format(
				top[0][0], len(model.AG.node[top[0][0]]['nodes']),
				top[0][1], len(model.AG.node[top[0][1]]['nodes']),
				top[1]['pval'], top[1]['strength']) + ' ' +
			        showAttributeNames(model, top[0], show_stars=show_stars) + '\n')

		f.write('\n')

		# Output TOP FREQUENT ASSOCIATIONS
		f.write('{0:6s} {1:6s} {2:8s} {3:6s}\n'.format('A1_siz', 'A2_siz', 'E[#]', 'Freq'))
		for top_frequent in top_frequents:
			attr1 = top_frequent[0][0]
			attr2 = top_frequent[0][1]
			freq = top_frequent[1]

			n1 = attr_counts[tuple(attr1)]
			n2 = attr_counts[tuple(attr2)]
			if attr1 == attr2:
				expected_edges = scipy.misc.comb(n1, 2) * d
			else:
				expected_edges = n1 * n2 * d
			f.write('{0:6n} {1:6n} {2:8.1f} {3:6n} '.format(n1, n2, expected_edges, freq))
			f.write(str(list(np.array(model.G.name)[np.array(attr1, dtype=bool)])) + '--' + str(
				list(np.array(model.G.name)[np.array(attr2, dtype=bool)])) + '\n')

		# Output Graph Patterns (e.g., stars, triangles, links, cliques)
		if show_graph_patterns:
			f.write('\n')
			f.write(get_graph_patterns(model, show_stars=show_stars, remove_homophily=remove_homophily))

		f.write('\n')

		# Output Significant associations minus frequnet associations
		f.write('Pattern difference\n')
		diff_patterns = _get_pattern_differences(model, tops, top_frequents)
		f.write('{0:4s} {1:4s}({2:3s}) {3:4s}({4:3s}) {5:3s} {6:3s}\n'.format('Rank','C1', 'siz', 'C2', 'siz', 'pvl', 'str'))
		for rank, top in diff_patterns:
			f.write('{0:3d} {1:4d}({2:3d}) {3:4d}({4:3d}) {5:0.3f} {6:3d}'.format(
				rank,
				top[0][0], len(model.AG.node[top[0][0]]['nodes']),
				top[0][1], len(model.AG.node[top[0][1]]['nodes']),
				top[1]['pval'], top[1]['strength']) + ' ' +
				showAttributeNames(model, top[0], show_stars=show_stars) + '\n')

def _get_significant_patterns(model, remove_homophily):
	interactions = {}
	for u, v in model.AG.edges():
		n1 = len(model.AG.node[u]['nodes'])
		n2 = len(model.AG.node[v]['nodes'])

		# Skip small clusters
		cluster_size_threshold = model.SUPPORT * model.G.number_of_nodes()
		if n1 < cluster_size_threshold or n2 < cluster_size_threshold:
			continue

		# Skip homophily relations
		if remove_homophily:
			continue_flag = True
			ones_u = model.AG.node[u]['ones']
			ones_v = model.AG.node[v]['ones']
			for i in range(model.dim):
				if remove_homophily == 'weak':
					if (ones_u[i] > 0) and (ones_v[i] == 0):
						continue_flag = False
					elif (ones_u[i] == 0) and (ones_v[i] > 0):
						continue_flag = False
				else:
					if (ones_u[i] == n1) and (ones_v[i] != n2):
						continue_flag = False
					elif (ones_u[i] != n1) and (ones_v[i] == n2):
						continue_flag = False
			if continue_flag:
				continue

		pval = stats.binom_test(model.AG.edge[u][v]['weight'], n1 * n2, model.density, alternative='greater')
		strength = model.AG.edge[u][v]['weight']
		interactions[(u, v)] = dict()
		interactions[(u, v)]['pval'] = pval
		interactions[(u, v)]['strength'] = strength

	top = sorted(interactions.items(), key=lambda x: x[1]['pval'])
	# Return only significant interactions
	top = filter(lambda interaction: interaction[1]['pval'] < ALPHA_FOR_SIGNIFICANT_PATTERNS, top)
	return top

# get top-k significant connections
def get_significant_patterns(model, k, show_stars=True, remove_homophily=False):
	tops = _get_significant_patterns(model, remove_homophily)[:k]

	print '{0:4s}({1:3s}) {2:4s}({3:3s}) {4:3s} {5:3s}'.format('C1', 'siz', 'C2', 'siz', 'pvl', 'str')
	for top in tops:
		print '{0:4d}({1:3d}) {2:4d}({3:3d}) {4:0.3f} {5:3d}'.format(
			top[0][0], len(model.AG.node[top[0][0]]['nodes']),
			top[0][1], len(model.AG.node[top[0][1]]['nodes']),
			top[1]['pval'], top[1]['strength']), showAttributeNames(model, top[0], show_stars=show_stars)


def _get_frequent_patterns(model, remove_homophily):
	pairs = defaultdict(int)

	for u, v in model.G.edges():
		# Skip homophily relations
		if remove_homophily and tuple(model.G.node[u]['attrs']) == tuple(model.G.node[v]['attrs']):
			continue

		attr_key = (tuple(model.G.node[u]['attrs']), tuple(model.G.node[v]['attrs']))
		if attr_key not in pairs:
			attr_key = (tuple(model.G.node[v]['attrs']), tuple(model.G.node[u]['attrs']))
		pairs[attr_key] += 1

	top = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

	return top

# get top-k frequent connections
def get_frequent_patterns(model, k, remove_homophily=False):
	tops = _get_frequent_patterns(model, remove_homophily)[:k]
	attrs = [tuple(model.G.node[node_id]['attrs']) for node_id in model.G.nodes()]
	attr_counts = Counter(attrs)
	d = nx.density(model.G)

	print '{0:6s} {1:6s} {2:8s} {3:6s}'.format('A1_siz', 'A2_siz', 'E[#]', 'Freq')
	for top in tops:
		attr1 = top[0][0]
		attr2 = top[0][1]
		freq = top[1]

		n1 = attr_counts[tuple(attr1)]
		n2 = attr_counts[tuple(attr2)]
		if attr1 == attr2:
			expected_edges = scipy.misc.comb(n1, 2) * d
		else:
			expected_edges = n1 * n2 * d

		print '{0:6n} {1:6n} {2:8.1f} {3:6n}'.format(n1, n2, expected_edges, freq),
		print np.array(model.G.name)[np.array(attr1, dtype=bool)], '--', np.array(model.G.name)[
			np.array(attr2, dtype=bool)]

# get the average of strengths of top connections
def getAvgTop(model, prob):
	k = int(model.AG.number_of_edges() * prob)
	if k < 1:
		return -1
	toplist = model.findTopStrongConn(k)
	total = 0
	for i in toplist:
		total += i[1]

	return total / k

def checkEqual(attr1, attr2, idx):
	for i in idx:
		if attr1[i] != attr2[i]:
			return False

	return True

def getNumOfInteractions(model, attr1, attr2):
	skipIndex = []
	for i in range(len(attr1)):
		if attr1[i] == -1 or attr2[i] == -1:
			skipIndex.append(i)

	idx = [i for i in range(len(attr1)) if i not in skipIndex]

	matchNum = 0
	for [u, v] in model.G.edges():
		temp = model.G.node[u]['attrs']
		if model.checkEqual(temp, attr1, idx) == True:
			if model.checkEqual(model.G.node[v]['attrs'], attr2, idx):
				matchNum += 1
		elif model.checkEqual(model.G.node[u]['attrs'], attr2, idx):
			if model.checkEqual(model.G.node[v]['attrs'], attr1, idx):
				matchNum += 1

	return matchNum


def extractDominantInteractions(model, k):
	dominantInter = {}
	for [u, v] in model.G.edges():
		concatAttrs = str(model.G.node[u]['attrs'] + model.G.node[v]['attrs'])
		if dominantInter.has_key(concatAttrs):
			dominantInter[concatAttrs] += 1
		elif dominantInter.has_key(str(model.G.node[v]['attrs'] + model.G.node[u]['attrs'])):
			dominantInter[str(model.G.node[v]['attrs'] + model.G.node[u]['attrs'])] += 1
		else:
			dominantInter[concatAttrs] = 1

	top = sorted(dominantInter.items(), key=lambda x: x[1])
	top.reverse()

	return top[0:k]


def showAttributeNames(model, interaction, show_stars):
	u, v = interaction
	return showAttributeName(model, u, show_stars) + '--' + showAttributeName(model, v, show_stars)

def showAttributeName(model, node_id, show_stars):
	u = node_id
	n1 = len(model.AG.node[u]['nodes'])
	ones_u = model.AG.node[u]['ones']
	touched_u = model.AG.node[u]['touched']

	if show_stars:
		attr1_str_list = []
		for i in range(model.dim):
			if ones_u[i] == n1:
				attr1_str_list.append(model.G.name[i])
			elif ones_u[i] > 0:
				p = get_conditional_probability(model, u, i)
				pval = stats.binom_test(ones_u[i], n1, p, alternative='two-sided')
				if pval < ALPHA_FOR_STAR_RESOLUTIONS:
					# Significant
					if p < float(ones_u[i])/n1:
						attr1_str_list.append(model.G.name[i] + '^')
				else:
					attr1_str_list.append('%s(%i)' % (model.G.name[i], ones_u[i]))
	else:
		attr1 = [0] * model.dim
		for i in range(model.dim):
			if ones_u[i] == n1:
				attr1[i] = 1

		attr1_str_list = list(np.array(model.G.name)[np.array(attr1, dtype=bool)])

	return str(attr1_str_list)

def _get_attribute_vector(model, node_id, show_stars=True):
	u = node_id
	n = len(model.AG.node[u]['nodes'])
	ones_u = model.AG.node[u]['ones']
	touched_u = model.AG.node[u]['touched']

	attr = [0] * model.dim
	if show_stars:
		for i in range(model.dim):
			if ones_u[i] == n:
				attr[i] = 1
			elif ones_u[i] > 0:
				p = get_conditional_probability(model, u, i)
				pval = stats.binom_test(ones_u[i], n, p, alternative='two-sided')
				if pval < ALPHA_FOR_STAR_RESOLUTIONS:
					# Significant
					if p < float(ones_u[i]) / n:
						attr[i] = 1
				else:
					attr[i] = -1
	else:
		attr = [0] * model.dim
		for i in range(model.dim):
			if ones_u[i] == n:
				attr[i] = 1
	return attr

def _convert_attribute_vector_to_name_list(model, attr_vector):
	name_list = []
	for i in range(model.dim):
		if attr_vector[i] == 1:
			name_list.append(model.G.name[i])
		elif attr_vector[i] == -1:
			name_list.append('%s(*)' % model.G.name[i])
	return name_list

def _get_pattern_differences(model, top_significants, top_frequents):
	r = []
	for idx, top_significant in enumerate(top_significants):
		node_id1 = top_significant[0][0]
		node_id2 = top_significant[0][1]
		attr1 = _get_attribute_vector(model, node_id1)
		attr2 = _get_attribute_vector(model, node_id2)

		is_remove = False
		for top_frequent in top_frequents:
			fattr1 = top_frequent[0][0]
			fattr2 = top_frequent[0][1]
			if sum(map(lambda x, y: x < 0 or x == y, attr1, fattr1)) == model.dim and \
							sum(map(lambda x, y: x < 0 or x == y, attr2, fattr2)) == model.dim:
				print 'removing', str(attr1), '--', str(attr2)
				print 'removing', str(_convert_attribute_vector_to_name_list(model, attr1)), '--', \
					str(_convert_attribute_vector_to_name_list(model, attr2))
				print 'matched ', str(fattr1), '--', str(fattr2)
				print 'matched ', str(_convert_attribute_vector_to_name_list(model, fattr1)), '--',\
					str(_convert_attribute_vector_to_name_list(model, fattr2))
				is_remove = True
				break
			if sum(map(lambda x, y: x < 0 or x == y, attr1, fattr2)) == model.dim and \
							sum(map(lambda x, y: x < 0 or x == y, attr2, fattr1)) == model.dim:
				print 'removing', str(attr1), '--', str(attr2)
				print 'removing', str(_convert_attribute_vector_to_name_list(model, attr1)), '--', \
					str(_convert_attribute_vector_to_name_list(model, attr2))
				print 'matched ', str(fattr2), '--', str(fattr1)
				print 'matched ', str(_convert_attribute_vector_to_name_list(model, fattr2)), '--', \
					str(_convert_attribute_vector_to_name_list(model, fattr1))
				is_remove = True
				break
		if is_remove == False:
			r.append((idx+1, top_significant))

	return r

# Find 2 stars and triangles
def get_graph_patterns(model, show_stars=True, remove_homophily=False):
	top_significant = _get_significant_patterns(model, remove_homophily)
	significant_edges = map(lambda p: p[0], top_significant)
	smallAG = nx.Graph(significant_edges)
	result_str = ''
	# import matplotlib.pyplot as plt
	# nx.draw(smallAG)
	# plt.show()
	triangles = []
	two_stars = []
	for node_id in smallAG.nodes():
		neighbors = smallAG.neighbors(node_id)
		if len(neighbors) >= 2:
			for i in range(len(neighbors)):
				for j in range(i + 1, len(neighbors)):
					if smallAG.has_edge(neighbors[i], neighbors[j]):
						# Triangle
						triangles.append(tuple(sorted([neighbors[i], node_id, neighbors[j]])))
					else:
						# 2-star
						two_stars.append(tuple([neighbors[i], node_id, neighbors[j]]))
	# uniquefy...
	triangles = set(triangles)

	result_str += '  {0:4s}({1:3s}) {2:4s}({3:3s}) {4:4s} ({5:3s})\n'.format('C1', 'siz', 'C2', 'siz', 'C3', 'siz')
	for t in triangles:
		result_str += 'T '
		result_str += '{0:4d}({1:3d}) {2:4d}({3:3d}) {4:4d}({5:3d}) '.format(t[0], len(model.AG.node[t[0]]['nodes']),
		                                                                     t[1], len(model.AG.node[t[1]]['nodes']),
		                                                                     t[2], len(model.AG.node[t[2]]['nodes']))
		result_str += '%s-%s-%s\n' % (showAttributeName(model, t[0], show_stars),
		                              showAttributeName(model, t[1], show_stars),
		                              showAttributeName(model, t[2], show_stars))
	result_str += '\n'
	for t in two_stars:
		result_str += 'C '
		result_str += '{0:4d}({1:3d}) {2:4d}({3:3d}) {4:4d}({5:3d}) '.format(t[0], len(model.AG.node[t[0]]['nodes']),
		                                                                     t[1], len(model.AG.node[t[1]]['nodes']),
		                                                                     t[2], len(model.AG.node[t[2]]['nodes']))
		result_str += '%s-%s-%s\n' % (showAttributeName(model, t[0], show_stars),
		                              showAttributeName(model, t[1], show_stars),
		                              showAttributeName(model, t[2], show_stars))
	result_str += '\n'
	cliques = [clique for clique in nx.find_cliques(smallAG)]
	cliques = filter(lambda c: len(c) >= 3, cliques)
	if len(cliques) > 0:
		result_str += '# Cliques\n'
	for clique in sorted(cliques, key=len, reverse=True):
		for node in clique:
			result_str += '{0:4d}({1:3d}) '.format(node, len(model.AG.node[node]['nodes']))
		result_str += '\n'
		for node in clique:
			result_str += '{0:s} '.format(showAttributeName(model, node, show_stars))
		result_str += '\n'

	return result_str

def get_conditional_probability(model, cid, star_idx):
	touched_idx = np.where(model.AG.node[cid]['touched'] == 1)[0]
	cluster_size = len(model.AG.node[cid]['nodes'])
	touched_values = {}
	for i in touched_idx:
		touched_values[i] = 1 if model.AG.node[cid]['ones'][i] == cluster_size else 0

	total = 0
	cnt = 0
	for u in model.G.nodes():
		same = True
		for i in touched_idx:
			if model.G.node[u]['attrs'][i] != touched_values[i]:
				same = False
				break

		if same:
			total += 1
			if model.G.node[u]['attrs'][star_idx] == 1:
				cnt += 1

	return float(cnt)/total

"""
The script below is used for showing all intermediate significant associations.
It may contain many redundant codes that replicate ones above.
"""

def _get_all_significant_patterns(model, k, show_stars=True, remove_homophily=False):
	interactions = []
	for association in model._significant_associations:
		n1, ones_u, touched_u = association[0]
		n2, ones_v, touched_v = association[1]
		pval = association[2]

		# Skip small clusters
		cluster_size_threshold = model.SUPPORT * model.G.number_of_nodes()
		if n1 < cluster_size_threshold or n2 < cluster_size_threshold:
			continue

		# Skip homophily relations
		if remove_homophily:
			continue_flag = True
			for i in range(model.dim):
				if remove_homophily == 'weak':
					if (ones_u[i] > 0) and (ones_v[i] == 0):
						continue_flag = False
					elif (ones_u[i] == 0) and (ones_v[i] > 0):
						continue_flag = False
				else:
					if (ones_u[i] == n1) and (ones_v[i] != n2):
						continue_flag = False
					elif (ones_u[i] != n1) and (ones_v[i] == n2):
						continue_flag = False
			if continue_flag:
				continue

		if pval < ALPHA_FOR_SIGNIFICANT_PATTERNS:
			interactions.append(association)

	tops = sorted(interactions, key=lambda x: x[2])[:k]
	return tops

def get_all_significant_patterns(model, k, show_stars=True, remove_homophily=False):
	tops = _get_all_significant_patterns(model, k, show_stars, remove_homophily)

	return_str = ''
	return_str += '{0:4s}({1:3s}) {2:4s}({3:3s}) {4:3s} {5:3s}\n'.format('C1', 'siz', 'C2', 'siz', 'pvl', 'str')
	for top in tops:
		return_str += '{0:4d}({1:3d}) {2:4d}({3:3d}) {4:0.3f} {5:3d} '.format(0, top[0][0], 0, top[1][0], top[2], top[3])
		return_str += showPartialAttributeNames(model, top, show_stars=show_stars)
		return_str += '\n'
	return return_str

def showPartialAttributeNames(model, association, show_stars):
	return showPartialAttributeName(model, association[0], show_stars) + \
			'--' + showPartialAttributeName(model, association[1], show_stars)

def showPartialAttributeName(model, cluster, show_stars):
	n1 = cluster[0]
	ones_u = cluster[1]

	if show_stars:
		attr1_str_list = []
		for i in range(model.dim):
			if ones_u[i] == n1:
				attr1_str_list.append(model.G.name[i])
			elif ones_u[i] > 0:
				p = get_conditional_probability_in_intermediate(model, cluster, i)
				pval = stats.binom_test(ones_u[i], n1, p, alternative='two-sided')
				if pval < ALPHA_FOR_STAR_RESOLUTIONS:
					# Significant
					if p < float(ones_u[i])/n1:
						attr1_str_list.append(model.G.name[i] + '^')
				else:
					attr1_str_list.append('%s(%i)' % (model.G.name[i], ones_u[i]))
	else:
		attr1 = [0] * model.dim
		for i in range(model.dim):
			if ones_u[i] == n1:
				attr1[i] = 1

		attr1_str_list = list(np.array(model.G.name)[np.array(attr1, dtype=bool)])

	return str(attr1_str_list)

def get_conditional_probability_in_intermediate(model, cluster, star_idx):
	touched_idx = np.where(cluster[2] == 1)[0]
	cluster_size = cluster[0]
	touched_values = {}
	for i in touched_idx:
		touched_values[i] = 1 if cluster[1][i] == cluster_size else 0

	total = 0
	cnt = 0
	for u in model.G.nodes():
		same = True
		for i in touched_idx:
			if model.G.node[u]['attrs'][i] != touched_values[i]:
				same = False
				break

		if same:
			total += 1
			if model.G.node[u]['attrs'][star_idx] == 1:
				cnt += 1

	return float(cnt)/total
