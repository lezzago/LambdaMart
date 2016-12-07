from lambdamart import LambdaMART
import numpy as np

def get_data(file_loc):
	f = open(file_loc, 'r')
	data = []
	for line in f:
		new_arr = []
		arr = line.split(' #')[0].split()
		score = arr[0]
		q_id = arr[1].split(':')[1]
		new_arr.append(int(score))
		new_arr.append(int(q_id))
		arr = arr[2:]
		for el in arr:
			new_arr.append(float(el.split(':')[1]))
		data.append(new_arr)
	f.close()
	return np.array(data)

training_data = get_data('/Users/madhavagrawal/Downloads/MQ2007/Fold1/train.txt')
test_data = get_data('/Users/madhavagrawal/Downloads/MQ2007/Fold1/test.txt')
model = LambdaMART(training_data, 500, 10, 0.001)
model.fit()
average_ndcg, predicted_scores = model.predict(test_data)
print average_ndcg