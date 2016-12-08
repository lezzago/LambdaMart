from lambdamart import LambdaMART
import numpy as np
import pandas as pd

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

def group_queries(data):
	query_indexes = {}
	index = 0
	for record in data:
		query_indexes.setdefault(record[1], [])
		query_indexes[record[1]].append(index)
		index += 1
	return query_indexes

training_data = get_data('/Users/madhavagrawal/Downloads/MQ2007/Fold1/train.txt')
test_data = get_data('/Users/madhavagrawal/Downloads/MQ2007/Fold1/test.txt')
# model = LambdaMART(training_data, 2, 10, 0.001)
# model.fit()
# model.save('lambdamart_model')
model = LambdaMART()
model.load('lambdamart_model.lmart')
average_ndcg, predicted_scores = model.validate(test_data)

print 'NDCG score: %f' % (average_ndcg)
query_indexes = group_queries(test_data)
index = query_indexes.keys()[0]
testdata = [test_data[i][0] for i in query_indexes[index]]
pred = [predicted_scores[i] for i in query_indexes[index]]
output = pd.DataFrame({"True label": testdata, "prediction": pred})
output = output.sort('prediction',ascending = False)
output.to_csv("outdemo.csv", index =False)
print output
# for i in query_indexes[index]:
# 	print test_data[i][0], predicted_scores[i]




