import numpy as np

def dcg(scores):
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 1)
						for i in xrange(len(scores) + 1)
					])

def compute_lambda()

class LambdaMART:

	def __init__(self, training_data, number_of_trees, leaves_per_tree, learning_rate):
		'''
		The format for training data is as follows:
			[relevance, q_id, [feature vector]]
		'''
		self.training_data = training_data
		self.number_of_trees = number_of_trees
		self.leaves_per_tree = self.leaves_per_tree
		self.learning_rate = learning_rate

	def fit(self):

		for k in self.number_of_trees:
			for i in xrange(len(training_data)):


def main():
	

if __name__ == '__main__':
	main()