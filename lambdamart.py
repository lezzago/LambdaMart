import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from RegressionTree import RegressionTree
from essemble import essemble_trees
import pandas as pd
import pickle

def dcg(scores):
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in xrange(len(scores))
					])
	# total = 0
	# for i in xrange(len(scores)):
	# 	total += (np.power(2.0, scores[i - 1]) - 1.0) / np.log2(i + 1)
	# return total

def ideal_dcg(scores):
	scores = [score for score in sorted(scores)[::-1]]
	return dcg(scores)

def compare_arr(a, b):
	if len(a) != len(b):
		return False
	for i in xrange(len(a)):
		if a[i] != b[i]:
			return False
	return True

def single_dcg(scores, i, j):
	return (np.power(2, scores[i]) - 1) / np.log2(j + 2)

def delta_ndcg(scores, i, j, idcg):

	dcg_val = dcg(scores)
	temp_scores = copy.deepcopy(scores)
	temp_scores[i], temp_scores[j] = temp_scores[j], temp_scores[i]
	new_dcg_val = dcg(temp_scores)

	# ideal_scores = [score for score in sorted(temp_scores) if score > 0]

	# ideal_dcg = dcg(ideal_scores)
	return abs(dcg_val - new_dcg_val)/idcg

# useless parallel code
def lambda_parallel(args):
	true_scores, good_ij_pairs, predicted_scores = args
	# print scores
	# exit()

	# true_scores, predicted_scores = scores
	i, j = good_ij_pairs
	z_ndcg = delta_ndcg(true_scores, i, j)
	rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
	rho_complement = 1.0 - rho
	lambda_val = z_ndcg * rho
	# lambdas = []
	# lambdas.append(lambda_val)
	# lambdas.append(-lambda_val)
	# w = []
	w_val = rho * rho_complement * z_ndcg
	# w.append(w_val)
	# w.append(w_val)
	return i, j, lambda_val, w_val


#true_scores, predicted_scores
def compute_lambda(args):
	true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args
	# true_scores, predicted_scores, good_ij_pairs = scores
	num_docs = len(true_scores)
	sorted_indexes = np.argsort(predicted_scores)[::-1]
	rev_indexes = np.argsort(sorted_indexes)
	true_scores = true_scores[sorted_indexes]
	predicted_scores = predicted_scores[sorted_indexes]

	lambdas = np.zeros(num_docs)
	w = np.zeros(num_docs)

	single_dcgs = {}
	for i,j in good_ij_pairs:
		if (i,i) not in single_dcgs:
			single_dcgs[(i,i)] = single_dcg(true_scores, i, i)
		single_dcgs[(i,j)] = single_dcg(true_scores, i, j)
		if (j,j) not in single_dcgs:
			single_dcgs[(j,j)] = single_dcg(true_scores, j, j)
		single_dcgs[(j,i)] = single_dcg(true_scores, j, i)
	### parallel portion
	# pool = Pool(processes=8)
	# for i, j, lambda_val, w_val in pool.map(lambda_parallel, zip([true_scores for i in xrange(len(good_ij_pairs))], good_ij_pairs, [predicted_scores for i in xrange(len(good_ij_pairs))])):
	# 	lambdas[i] += lambda_val
	# 	lambdas[j] -= lambda_val
	# 	w[i] += w_val
	# 	w[j] += w_val
	# pool.close()

	for i,j in good_ij_pairs:
		# z_ndcg = delta_ndcg(true_scores, i, j, idcg)
		z_ndcg = abs(single_dcgs[(i,j)] - single_dcgs[(i,i)] + single_dcgs[(j,i)] - single_dcgs[(j,j)]) / idcg
		rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
		rho_complement = 1.0 - rho
		lambda_val = z_ndcg * rho
		lambdas[i] += lambda_val
		lambdas[j] -= lambda_val

		w_val = rho * rho_complement * z_ndcg
		w[i] += w_val
		w[j] += w_val

	return lambdas[rev_indexes], w[rev_indexes], query_key

def group_queries(training_data):
	query_indexes = {}
	index = 0
	for record in training_data:
		query_indexes.setdefault(record[1], [])
		query_indexes[record[1]].append(index)
		index += 1
	return query_indexes

def get_pairs(scores):
	query_pair = []
	for query_scores in scores:
		temp = sorted(query_scores, reverse=True)
		pairs = []
		for i in xrange(len(temp)):
			for j in xrange(len(temp)):
				if temp[i] > temp[j]:
					pairs.append((i,j))
		query_pair.append(pairs)
	return query_pair

class LambdaMART:

	def __init__(self, training_data=None, number_of_trees=0, leaves_per_tree=0, learning_rate=0):
		'''
		The format for training data is as follows:
			[relevance, q_id, [feature vector]]
		'''
		self.training_data = np.array(training_data)
		self.number_of_trees = number_of_trees
		self.leaves_per_tree = leaves_per_tree
		self.learning_rate = learning_rate
		self.trees = []

	def fit(self):
		predicted_scores = np.zeros(len(self.training_data))
		query_indexes = group_queries(self.training_data)
		query_keys = query_indexes.keys()
		true_scores = [self.training_data[query_indexes[query], 0] for query in query_keys]
		good_ij_pairs = get_pairs(true_scores)
		tree_data = pd.DataFrame(self.training_data[:, 2:7])
		labels = self.training_data[:, 0]



		# ideal dcg calculation
		idcg = [ideal_dcg(scores) for scores in true_scores]

		for k in xrange(self.number_of_trees):
			print 'Tree %d' % (k)
			lambdas = np.zeros(len(predicted_scores))
			w = np.zeros(len(predicted_scores))
			pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]
			
			pool = Pool()
			for lambda_val, w_val, query_key in pool.map(compute_lambda, zip(true_scores, pred_scores, good_ij_pairs, idcg, query_keys), chunksize=1):
				indexes = query_indexes[query_key]
				lambdas[indexes] = lambda_val
				w[indexes] = w_val
			pool.close()

			## non parallel
			# for i in xrange(len(true_scores)):
			# 	query = query_keys[i]
			# 	lambdas[query_indexes[query]], w[query_indexes[query]] = compute_lambda(true_scores[i], pred_scores[i], good_ij_pairs[i])

			# sklearn tree			
			tree = DecisionTreeRegressor(max_depth=50)
			tree.fit(self.training_data[:,2:], lambdas)
			self.trees.append(tree)
			prediction = tree.predict(self.training_data[:,2:])
			predicted_scores += prediction * self.learning_rate
			# tree = RegressionTree(tree_data, lambdas, max_depth=10, ideal_ls= 0.001)
			# print 'created tree'
			# tree.fit()
			# print 'fitted tree'
			# prediction = tree.predict(self.training_data[:,2:])
			# print 'predicted tree'
			# print prediction
			# # exit()
			# predicted_scores = essemble_trees(lambdas, w, prediction, predicted_scores, self.learning_rate)
			# print 'updates scores'

			###
			# Tree code here, already calculated lambdas and ws
			###

		# print predicted_scores


	def predict(self, data):
		data = np.array(data)
		query_indexes = group_queries(data)
		average_ndcg = 0
		predicted_scores = np.zeros(len(data))
		for query in query_indexes:
			results = np.zeros(len(query_indexes[query]))
			for tree in self.trees:
				results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
			predicted_scores[query_indexes[query]] = results
			ndcg_val = (dcg(results) / ideal_dcg(results))
			average_ndcg += ndcg_val
		average_ndcg /= len(query_indexes)
		return average_ndcg, predicted_scores

	def save(self, fname):
		pickle.dump(self, open('%s.lmart' % (fname), "w"), protocol=2)

	def load(self, fname):
		model = pickle.load(open(fname , "r"))
		self.training_data = model.training_data
		self.number_of_trees = model.number_of_trees
		self.leaves_per_tree = model.leaves_per_tree
		self.learning_rate = model.learning_rate
		self.trees = model.trees

def main():
	f = open('vali.txt', 'r')
	count = 0
	training_data = []
	test_data = []
	for line in f:
		if count >= 240:
			break
		new_arr = []
		arr = line.split(' #')[0].split()
		score = arr[0]
		q_id = arr[1].split(':')[1]
		new_arr.append(int(score))
		new_arr.append(int(q_id))
		arr = arr[2:]
		for el in arr:
			new_arr.append(float(el.split(':')[1]))
		if count < 200:
			training_data.append(new_arr)
		else:
			test_data.append(new_arr)
		count += 1
	f.close()
	model = LambdaMART(training_data, 20, 10, 0.001)
	model.fit()
	model.save('temp')
	t_model = LambdaMART()
	t_model.load('temp.lmart')
	average_ndcg, predicted_scores = t_model.predict(test_data)
	print average_ndcg




if __name__ == '__main__':
	main()