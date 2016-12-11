import networkx as nx
import random
import numpy as np
from scipy import stats
import sys
import csv
from collections import defaultdict

import data.reader as reader
import algorithm.transform as tr
from algorithm import analyze

class Link_Prediction:
	RESOURCE_ALLOCATION = 'resource_allocation'
	JACCARD_COEFFICIENT = 'jaccard_coefficient'
	ADAMIC_ADAR_INDEX = 'adamic_adar_index'
	PREFERENTIAL_ATTACHMENT = 'preferential_attachment'

	def __init__(self, old_G, new_G, model):
		self._old_G = old_G
		self._new_G = new_G
		self.new_links = None
		self._model = model
		self._preds = {}

		self._max_significance = {}
		# normalize significance of all associations such that it is in range of 0 ~ 1.
		print 'Normalizing signifincance ...'
		self._normalize_significance()
		print 'Normalizing frequency...'
		self._normalize_frequency()
		# get attributes for each cluster while considering significance of stars
		print 'getting attributes for each cluster ...'
		self._get_attributes()
		# find the core
		print 'finding core ...'
		self._find_core()
		# get new links that are newly created in the new graph
		print 'getting newly created links ...'
		self._get_new_links()
		print 'negative sampling ...'
		self._negative_sampling()

	def _negative_sampling(self):
		self._neg_samples = []

		for u, v in self.new_links:
			# for each u, randomly samples 5 negative pairs
			samples = random.sample(self._old_G.nodes(), 100)
			random.shuffle(samples)
			i = 0
			neighbors = nx.neighbors(self._old_G, u)
			for s in samples:
				if s not in neighbors and s != v and (u, s) not in self._neg_samples:
					self._neg_samples.append((u, s))
					i += 1
					if i == 1:
						break

	def _find_core(self):
		self._core = []
		for u in self._old_G.nodes():
			if len(self._old_G[u]) >= 3 and len(self._new_G[u]) >= 3:
				self._core.append(u)

	def _normalize_significance(self):
		min_pval = 0.01
		max_pval = 0

		for (u, v) in self._model.AG.edges():
			# calculate the p-value of each association
			n1 = len(self._model.AG.node[u]['nodes'])
			n2 = len(self._model.AG.node[v]['nodes'])
			pval = stats.binom_test(self._model.AG.edge[u][v]['weight'], n1 * n2, self._model.density, alternative='greater')
			if pval >= 0.01:
				pval = 0.01
			self._model.AG.edge[u][v]['pval'] = pval

			# update the min p-value if the current association has the p-value lower than the current min p-value
			if pval < min_pval:
				min_pval = pval
			# update the max p-value if the current association has the p-value greater than the current max p-value
			if pval > max_pval:
				max_pval = pval

		for (u, v) in self._model.AG.edges():
			# normalize the significance
			normalized = (self._model.AG.edge[u][v]['pval'] - min_pval) / (max_pval - min_pval)
			self._model.AG.edge[u][v]['significance'] = 1 - normalized

	def _normalize_frequency(self):
		frequency = defaultdict(int)
		for (u, v) in self._old_G.edges():
			if u <= v:
				attr_u = self._model.G.node[u]['attrs']
				attr_v = self._model.G.node[v]['attrs']
				pattern = (tuple(attr_u), tuple(attr_v))
				if pattern in frequency:
					frequency[pattern] += 1
				else:
					pattern = (tuple(attr_v), tuple(attr_u))
					frequency[pattern] += 1

		max_freq = 0
		min_freq = sys.maxint
		for pattern in frequency:
			if frequency[pattern] > max_freq:
				max_freq = frequency[pattern]
			if frequency[pattern] < min_freq:
				min_freq = frequency[pattern]
		diff = max_freq - min_freq
		for pattern in frequency:
			frequency[pattern] = (frequency[pattern] - min_freq) / float(diff)

		self._frequency = frequency

	def _get_frequency(self, u, v):
		attr_u = self._model.G.node[u]['attrs']
		attr_v = self._model.G.node[v]['attrs']
		if u <= v:
			pattern = (tuple(attr_u), tuple(attr_v))
		else:
			pattern = (tuple(attr_v), tuple(attr_u))

		return self._frequency[pattern]

	def _get_attributes(self):
		self._attributes = {}
		for c in self._model.AG.nodes():
			n = len(self._model.AG.node[c]['nodes'])
			ones = self._model.AG.node[c]['ones']
			attrs = [0] * self._model.dim
			for i in range(self._model.dim):
				if ones[i] == n:
					attrs[i] = 1
				elif ones[i] > 0:
					p = analyze.get_conditional_probability(self._model, c, i)
					pval = stats.binom_test(ones[i], n, p, alternative='two-sided')
					if pval < analyze.ALPHA_FOR_STAR_RESOLUTIONS:
						# Significant
						if p < float(ones[i])/n:
							attrs[i] = 1
						else:
							attrs[i] = 0
					else:
						attrs[i] = -1

			self._attributes[c] = attrs

	def _match_attributes(self, c, attr):
		match = True
		for i in range(len(self._attributes[c])):
			#if self._attributes[c][i] != -1 and self._attributes[c][i] != attr[i]:
			if self._attributes[c][i] != attr[i]:
				match = False
				break

		return match

	def _get_new_links(self):
		# set of edges of new graph
		edges_in_new_G = self._new_G.edges()
		# set of edges of old graph
		edges_in_old_G = self._old_G.edges()

		# find a set of links that appear in the new graph but are not present in the old graph
		self.new_links = []
		for u, v in list(set(edges_in_new_G) - set(edges_in_old_G)):
			# store only links whose both of two end nodes are present in the old graph

			#if u in self._core and v in self._core:
			if u in self._old_G and v in self._old_G:
				self.new_links.append((u, v))

	def _get_significance(self, u, v):
		attr_u = self._model.G.node[u]['attrs']
		attr_v = self._model.G.node[v]['attrs']
		pattern = (tuple(attr_u), tuple(attr_v))

		if pattern in self._max_significance:
			return self._max_significance[pattern]
		else:
			max_significance = 0
			for (c1, c2) in self._model.AG.edges():
				if (self._match_attributes(c1, attr_u) and self._match_attributes(c2, attr_v)) \
					or (self._match_attributes(c2, attr_u) and self._match_attributes(c1, attr_v)):
					if self._model.AG.edge[c1][c2]['significance'] > max_significance:
						max_significance = self._model.AG.edge[c1][c2]['significance']
			self._max_significance[pattern] = max_significance

		return self._max_significance[pattern]

	def _do_raw_prediction(self, method):
		# get the set of pairs of nodes that do not have an edge between them
		no_edges = []
		no_edges.extend(self.new_links)
		no_edges.extend(self._neg_samples)

		# perform prediction for each of the edges in the old graph
		# according to the input method, call corresponding link prediction algorithm
		print 'predicting links ...'
		if method == self.RESOURCE_ALLOCATION:
			preds = nx.resource_allocation_index(self._old_G, no_edges)
		elif method == self.JACCARD_COEFFICIENT:
			preds = nx.jaccard_coefficient(self._old_G, no_edges)
		elif method == self.ADAMIC_ADAR_INDEX:
			preds = nx.adamic_adar_index(self._old_G, no_edges)
		elif method == self.PREFERENTIAL_ATTACHMENT:
			preds = nx.preferential_attachment(self._old_G, no_edges)
		else:
			# if the input method is wrong, then show error message and terminate program
			print 'ERROR: wrong method name'
			print 'should be one of ("resource_allocation", "jaccard_coefficient",' \
				  ' "adamic_adar_index", "preferential_attachment")'
			sys.exit()

		print 'prediction is done ...'

		self._preds[method] = preds

	def perform_prediction(self, method, alpha=0.5):
		if method not in self._preds:
			self._do_raw_prediction(method)
		preds = self._preds[method]
		# get the number of new links
		number_of_new_links = len(self.new_links)

		self.predictions = []
		self.combined_predictions = []
		# compute combined scores (\alpha * confidence + (1 - \alpha) * significance)
		confidence_preds = []
		combined_preds = []
		a = str(int(alpha*10))
		writer = csv.writer(open('experiment/link_prediction/prediction_result_'+a+'.csv', 'w'), delimiter=',')
		writer.writerow(['node_1', 'node_2', 'isNewLink', 'jaccard', 'significance', 'jacc+sig', 'frequency', 'jacc+freq'])
		pos_results = []
		neg_results = []

		i = 0
		for (u, v, p) in preds:
			if i % 100 == 0:
				print i, 'th prediction processed'
			i+=1

			significance = self._get_significance(u, v)
			frequency = self._get_frequency(u, v)
			new_score_sig = alpha * p + (1 - alpha) * significance
			new_score_freq = alpha * p + (1 - alpha) * frequency
			if (u, v) in self.new_links or (v, u) in self.new_links:
				pos_results.append([u, v, True, p, significance, new_score_sig, frequency, new_score_freq])
			else:
				neg_results.append([u, v, False, p, significance, new_score_sig, frequency, new_score_freq])

		pos_results = sorted(pos_results, key=lambda x:x[4], reverse=True)
		neg_results = sorted(neg_results, key=lambda x:x[4], reverse=True)
		for row in pos_results:
			writer.writerow(row)
		for row in neg_results:
			writer.writerow(row)

	def _get_attr_names(self, u, v):
		print np.array(self._old_G.name)[np.array(self._old_G.node[u]['attrs'], dtype=bool)], '--', np.array(self._old_G.name)[np.array(self._old_G.node[v]['attrs'], dtype=bool)]

	def _check_patterns(self, model, u, v, top_significant, top_frequent):
		significant = False
		frequent = False

		for pattern in top_significant:
			c1, c2 = pattern[0]
			n1 = len(model.AG.node[c1]['nodes'])
			n2 = len(model.AG.node[c2]['nodes'])
			ones_u = model.AG.node[c1]['ones']
			ones_v = model.AG.node[c2]['ones']
			attr1 = [0] * model.dim
			attr2 = [0] * model.dim
			for i in range(model.dim):
				if ones_u[i] == n1:
					attr1[i] = 1
				if ones_v[i] == n2:
					attr2[i] = 1

			if (attr1 == self._old_G.node[u]['attrs'] and attr2 == self._old_G.node[v]['attrs']) or (attr1 == self._old_G.node[v]['attrs'] and attr2 == self._old_G.node[u]['attrs']):
				significant = True

		for pattern in top_frequent:
			if (list(pattern[0][0]) == self._old_G.node[u]['attrs'] and list(pattern[0][1]) == self._old_G.node[v]['attrs']) or (list(pattern[0][1]) == self._old_G.node[u]['attrs'] and list(pattern[0][0]) == self._old_G.node[v]['attrs']):
				frequent = True

		return significant, frequent

	def write_results(self, model):
		from algorithm import analyze
		top_significant = analyze._get_significant_patterns(model, 20)
		top_frequent = analyze._get_frequent_patterns(model, 20)

		significant_cnt = 0
		frequent_cnt = 0
		significant_cnt_hetero = 0
		frequent_cnt_hetero = 0
		only_significant_cnt = 0
		f = open('link_prediction_results.txt', 'w')
		for pred in self._predicted:
			u, v = pred[0]
			significant, frequent = self._check_patterns(model, u, v, top_significant, top_frequent)
			if significant:
				significant_cnt += 1
				if self._old_G.node[u]['attrs'] != self._old_G.node[v]['attrs']:
					significant_cnt_hetero += 1
			if frequent:
				frequent_cnt += 1
				if self._old_G.node[u]['attrs'] != self._old_G.node[v]['attrs']:
					frequent_cnt_hetero += 1
			if significant and not frequent:
				only_significant_cnt += 1

			f.write(str(round(pred[1], 2)) + ', ' + str(pred[2]) + ', ' + str(significant) + ', ' + str(frequent) + ', '
					+ str(list(np.array(self._old_G.name)[np.array(self._old_G.node[u]['attrs'], dtype=bool)])) + '--'
					+ str(list(np.array(self._old_G.name)[np.array(self._old_G.node[v]['attrs'], dtype=bool)]))+ '\n')

		f.write(str(significant_cnt) + '\t' + str(frequent_cnt) + '\t' + str(significant_cnt_hetero) + '\t' + str(frequent_cnt_hetero) + '\t' + str(only_significant_cnt))
		f.close()

def analyze_result(filepath):
	output_file = filepath.split('.')[0] + '_roc' + '.csv'
	writer = csv.writer(open(output_file, 'w'))
	writer.writerow(['threshold',
					 'jacc_tpr', 'jacc_fpr', 'jacc_prec', 'jacc_recall', 'jacc_fmeasure',
					 'jacc_sig_tpr', 'jacc_sig_fpr', 'jacc_sig_prec', 'jacc_sig_recall', 'jacc_sig_fmeasure',
					 'jacc_freq_tpr', 'jacc_freq_fpr', 'jacc_freq_prec', 'jacc_freq_recall', 'jacc_freq_fmeasure'])

	thresholds = np.linspace(0, 1, 1000)
	for threshold in thresholds:
		reader = csv.reader(open(filepath))
		reader.next()
		jacc_result = [0] * 4
		jacc_sig_result = [0] * 4
		jacc_freq_result = [0] * 4

		for row in reader:
			label = row[2]
			jacc = float(row[3])
			jacc_sig = float(row[5])
			jacc_freq = float(row[7])

			if label == 'True':
				if jacc >= threshold:
					# TP
					jacc_result[0] += 1
				else:
					# FN
					jacc_result[3] += 1

				if jacc_sig >= threshold:
					jacc_sig_result[0] += 1
				else:
					jacc_sig_result[3] += 1

				if jacc_freq >= threshold:
					jacc_freq_result[0] += 1
				else:
					jacc_freq_result[3] += 1
			else:
				if jacc >= threshold:
					# FP
					jacc_result[2] += 1
				else:
					# TN
					jacc_result[1] += 1

				if jacc_sig >= threshold:
					jacc_sig_result[2] += 1
				else:
					jacc_sig_result[1] += 1

				if jacc_freq >= threshold:
					jacc_freq_result[2] += 1
				else:
					jacc_freq_result[1] += 1

		beta = 2
		jacc_tpr = float(jacc_result[0]) / (jacc_result[0] + jacc_result[3])
		jacc_fpr = float(jacc_result[2]) / (jacc_result[2] + jacc_result[1])
		jacc_prec = float(jacc_result[0]) / (jacc_result[0] + jacc_result[2]) \
					if (jacc_result[0] + jacc_result[2]) > 0 else 0
		jacc_recall = float(jacc_result[0]) / (jacc_result[0] + jacc_result[3])\
					if (jacc_result[0] + jacc_result[3]) > 0 else 0
		jacc_fmeasure = f_measure(jacc_prec, jacc_recall, beta)

		jacc_sig_tpr = float(jacc_sig_result[0]) / (jacc_sig_result[0] + jacc_sig_result[3])
		jacc_sig_fpr = float(jacc_sig_result[2]) / (jacc_sig_result[2] + jacc_sig_result[1])
		jacc_sig_prec = float(jacc_sig_result[0]) / (jacc_sig_result[0] + jacc_sig_result[2])\
					if (jacc_sig_result[0] + jacc_sig_result[2]) > 0 else 0
		jacc_sig_recall = float(jacc_sig_result[0]) / (jacc_sig_result[0] + jacc_sig_result[3])\
					if (jacc_sig_result[0] + jacc_sig_result[3]) > 0 else 0
		jacc_sig_fmeasure = f_measure(jacc_sig_prec, jacc_sig_recall, beta)

		jacc_freq_tpr = float(jacc_freq_result[0]) / (jacc_freq_result[0] + jacc_freq_result[3])
		jacc_freq_fpr = float(jacc_freq_result[2]) / (jacc_freq_result[2] + jacc_freq_result[1])
		jacc_freq_prec = float(jacc_freq_result[0]) / (jacc_freq_result[0] + jacc_freq_result[2])\
					if (jacc_freq_result[0] + jacc_freq_result[2]) > 0 else 0
		jacc_freq_recall = float(jacc_freq_result[0]) / (jacc_freq_result[0] + jacc_freq_result[3])\
					if (jacc_freq_result[0] + jacc_freq_result[3]) > 0 else 0
		jacc_freq_fmeasure = f_measure(jacc_freq_prec, jacc_freq_recall, beta)

		writer.writerow([threshold,
						 jacc_tpr, jacc_fpr, jacc_prec, jacc_recall, jacc_fmeasure,
						 jacc_sig_tpr, jacc_sig_fpr, jacc_sig_prec, jacc_sig_recall, jacc_sig_fmeasure,
						 jacc_freq_tpr, jacc_freq_fpr, jacc_freq_prec, jacc_freq_recall, jacc_freq_fmeasure])

def f_measure(prec, recall, beta):
	if prec == 0 and recall == 0:
		return 0
	return (1 + beta ** 2) * ((prec * recall) / ((beta ** 2 * prec) + recall))

if __name__ == '__main__':
	'''
	import json
	import os
	os.chdir('../')
	configs = json.load(open('configuration/link_prediction.conf'))
	data_configs = configs['data']
	model_configs = configs['model']

	dr = reader.DatasetReader(data_configs)
	old_G = dr.load()
	data_configs['suffix'] = '2016'
	dr = reader.DatasetReader(data_configs)
	new_G = dr.load()

	gt = tr.GraphTrans(old_G, model_configs)
	gt.load_model(model_configs['AG_file'])

	lp = Link_Prediction(old_G, new_G, gt)
	lp.perform_prediction('jaccard_coefficient', 0.2)
	'''
	analyze_result('/Users/redrum21c/Dropbox/research/attribute_association/src/experiment/link_prediction/prediction_result_2.csv')
	analyze_result('/Users/redrum21c/Dropbox/research/attribute_association/src/experiment/link_prediction/prediction_result_5.csv')
	analyze_result('/Users/redrum21c/Dropbox/research/attribute_association/src/experiment/link_prediction/prediction_result_8.csv')
