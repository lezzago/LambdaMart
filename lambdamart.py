import numpy as np
import math
import random
import copy

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

def compute_lambda(true_scores, predicted_scores):
	num_docs = len(true_scores)
	sorted_indexes = np.argsort(predicted_scores)[::-1]
	rev_indexes = np.argsort(sorted_indexes)
	true_scores = true_scores[sorted_indexes]
	predicted_scores = predicted_scores[sorted_indexes]

	lambdas = np.zeros(num_docs)
	w = np.zeros(num_docs)
	temp = set([i for i in xrange(num_docs)])
	for i in xrange(num_docs):
		for j in xrange(num_docs):
			if true_scores[i] > true_scores[j]:
				if i in temp:
					temp.remove(i)
				if j in temp:
					temp.remove(j)
				z_ndcg = delta_ndcg(true_scores, i, j)
				rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
				rho_complement = 1.0 - rho
				lambda_val = z_ndcg * rho
				lambdas[i] += lambda_val
				lambdas[j] -= lambda_val

				w_val = rho * rho * z_ndcg
				w[i] += w_val
				w[j] -= w_val

	return lambdas[rev_indexes], w[rev_indexes]

def group_queries(training_data):
	query_indexes = {}
	index = 0
	for record in training_data:
		query_indexes.setdefault(record[1], [])
		query_indexes[record[1]].append(index)
		index += 1
	return query_indexes

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
		# training_data = np.array(training_data)
		# predicted_scores = np.array([random.random() for i in xrange(len(self.training_data))])
		predicted_scores = np.zeros(len(self.training_data))
		query_indexes = group_queries(self.training_data)
		for k in xrange(self.number_of_trees):
			lambdas = np.zeros(len(predicted_scores))
			w = np.zeros(len(predicted_scores))
			# for i in xrange(len(self.training_data)):
			for query in query_indexes:
				indexes = query_indexes[query]
				# print len(indexes)
				# print len(self.training_data[indexes]), len(self.training_data[indexes][1])
				# print self.training_data[0,1]
				lambdas[indexes], w[indexes] = compute_lambda(self.training_data[indexes, 0], predicted_scores[indexes])
			print w
			exit()

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