import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool

def dcg(scores):
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in xrange(len(scores))
					])
	# total = 0
	# for i in xrange(len(scores)):
	# 	total += (np.power(2.0, scores[i - 1]) - 1.0) / np.log2(i + 1)
	# return total

def compare_arr(a, b):
	if len(a) != len(b):
		return False
	for i in xrange(len(a)):
		if a[i] != b[i]:
			return False
	return True

def delta_ndcg(scores, i, j):

	dcg_val = dcg(scores)
	temp_scores = copy.deepcopy(scores)
	temp_scores[i], temp_scores[j] = temp_scores[j], temp_scores[i]
	new_dcg_val = dcg(temp_scores)

	ideal_scores = [score for score in sorted(temp_scores) if score > 0]

	ideal_dcg = dcg(ideal_scores)
	return abs(dcg_val - new_dcg_val)/ideal_dcg

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
	true_scores, predicted_scores, good_ij_pairs, query_key = args
	# true_scores, predicted_scores, good_ij_pairs = scores
	num_docs = len(true_scores)
	sorted_indexes = np.argsort(predicted_scores)[::-1]
	rev_indexes = np.argsort(sorted_indexes)
	true_scores = true_scores[sorted_indexes]
	predicted_scores = predicted_scores[sorted_indexes]

	lambdas = np.zeros(num_docs)
	w = np.zeros(num_docs)

	### parallel portion
	# pool = Pool(processes=8)
	# for i, j, lambda_val, w_val in pool.map(lambda_parallel, zip([true_scores for i in xrange(len(good_ij_pairs))], good_ij_pairs, [predicted_scores for i in xrange(len(good_ij_pairs))])):
	# 	lambdas[i] += lambda_val
	# 	lambdas[j] -= lambda_val
	# 	w[i] += w_val
	# 	w[j] += w_val
	# pool.close()

	for i,j in good_ij_pairs:
		z_ndcg = delta_ndcg(true_scores, i, j)
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

	def __init__(self, training_data, number_of_trees, leaves_per_tree, learning_rate):
		'''
		The format for training data is as follows:
			[relevance, q_id, [feature vector]]
		'''
		self.training_data = np.array(training_data)
		self.number_of_trees = number_of_trees
		self.leaves_per_tree = leaves_per_tree
		self.learning_rate = learning_rate

	def fit(self):
		predicted_scores = np.zeros(len(self.training_data))
		query_indexes = group_queries(self.training_data)
		query_keys = query_indexes.keys()
		true_scores = [self.training_data[query_indexes[query], 0] for query in query_keys]
		good_ij_pairs = get_pairs(true_scores)
		for k in xrange(self.number_of_trees):
			print 'Tree %d' % (k)
			lambdas = np.zeros(len(predicted_scores))
			w = np.zeros(len(predicted_scores))
			pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]
			
			pool = Pool(processes=4)
			for lambda_val, w_val, query_key in pool.map(compute_lambda, zip(true_scores, pred_scores, good_ij_pairs, query_keys)):
				indexes = query_indexes[query_key]
				lambdas[indexes] = lambda_val
				w[indexes] = w_val
			pool.close()

			## non parallel
			# for i in xrange(len(true_scores)):
			# 	query = query_keys[i]
			# 	lambdas[query_indexes[query]], w[query_indexes[query]] = compute_lambda(true_scores[i], pred_scores[i], good_ij_pairs[i])

			# sklearn tree			
			# tree = DecisionTreeRegressor(max_depth=6)
			# tree.fit(self.training_data[:,2:], lambdas)
			# prediction = tree.predict(self.training_data[:,2:])
			# predicted_scores += prediction * self.learning_rate

			###
			# Tree code here, already calculated lambdas and ws
			###

		print predicted_scores
	# def predict(self, data):


def main():
	f = open('vali.txt', 'r')
	count = 0
	training_data = []
	for line in f:
		if count > 200:
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
		training_data.append(new_arr)
		count += 1
	f.close()
	model = LambdaMART(training_data, 200, 10, 0.1)
	model.fit()


if __name__ == '__main__':
	main()