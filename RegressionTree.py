# regression tree
# input is a dataframe of features
# the corresponding y value(called labels here) is the scores for each document

import pandas as pd 
import numpy as np 

node_id = 0

def get_splitting_points(attribute):
	# given a list
	# return a list of possible splitting values
	attribute.sort()
	possible_split = []
	for i in range(len(attribute)-1):
		if attribute[i] != attribute[i+1]:
			possible_split.append(np.mean((attribute[i],attribute[i+1])))
	return possible_split

# create a dictionary, key is the attribute number, value is whole list of possible splits for that column

def find_best_split(data, label, split_points): 
	# split_points is a dictionary of possible splitting values
	# return the best split
	best_ls = 1000000
	best_split = None
	for key,possible_split in split_points.items():
		for split in possible_split:
			children = split_children(data, label, key, split)

			#weighted average of left and right ls
			ls  = float(len(children[1]))/len(label)*least_square(children[1]) + float(len(children[3]))/len(label)*least_square(children[3])
			if ls < best_ls:
				best_ls = ls
				best_split = (key, split)
			


	return best_split # return a tuple(attribute, value)

def split_children(data, label, key, split):
	left_index = [index for index in range(len(data.iloc[:,key])) if data.iloc[index,key] < split]
	right_index = [index for index in range(len(data.iloc[:,key])) if data.iloc[index,key] >= split]
	left_data = data.iloc[left_index,:]
	right_data = data.iloc[right_index,:]
	left_label = [label[i] for i in left_index]
	right_label =[label[i] for i in right_index]
	
	return left_data, left_label, right_data, right_label 

def least_square(label):
	# given the 
	# return the
	label = np.array(label).astype(float)
	return np.sum((label - np.mean(label))**2)


def create_leaf(label):
	global node_id
	node_id += 1
	leaf = {'splittng_feature': None,
			'left': None,
			'right':None,
			'is_leaf':True,
			'index':node_id}
	leaf['value'] = round(np.mean(label),3)
	return leaf


def create_tree(data, all_pos_split, label, current_depth = 0, max_depth = 5, ideal_ls = 100):
	
	remaining_features = all_pos_split
	#stopping conditions
	if sum([len(v)!= 0 for v in remaining_features.values()]) == 0:
		print current_depth
		print "stop because no more features."    
		# If there are no remaining features to consider, make current node a leaf node
		return create_leaf(label)    
	# #Additional stopping condition (limit tree depth)
	elif current_depth > max_depth:
		print current_depth
		print "reached max depth, stop."
		return create_leaf(label)

	# find splitting features    
	splitting_feature = find_best_split(data, label, remaining_features)  #tuple(id, value)
	# print "split feature: ", splitting_feature
	# remove current features
	remaining_features[splitting_feature[0]].remove(splitting_feature[1])
	#split on the features we want
	children = split_children(data, label, splitting_feature[0], splitting_feature[1])
	left_data = children[0]
	left_label =children[1]
	right_data = children[2]
	right_label = children[3]


	# Create a leaf node if the split is "perfect"
	if least_square(left_label) < ideal_ls:
		# print "current left ls: ", least_square(left_label), left_label
		print current_depth
		print "left stop because less than ls ."
		return create_leaf(left_label)
	if least_square(right_label) < ideal_ls:
		# print "right ls: ", least_square(right_label),right_label
		print current_depth
		print "right stop becasue less than ls."
		return create_leaf(right_label)

	# recurse on children
	left_tree = create_tree(left_data, remaining_features, left_label,  current_depth +1, max_depth , ideal_ls)
	right_tree = create_tree(right_data, remaining_features, right_label,  current_depth +1, max_depth , ideal_ls)
	return {'is_leaf'          : False, 
			'value'       : None,
			'splitting_feature': splitting_feature,
			'left'             : left_tree, 
			'right'            : right_tree,
			'index'			   : None}

def make_prediction(tree, x, annotate = False):
	if tree['is_leaf']:
		if annotate: 
			print "At leaf, predicting %s" % tree['value']
		return tree['index'], tree['value'] 
	else:
		# the splitting value of x.
		split_feature_value = x[tree['splitting_feature'][0]]
		if annotate: 
			print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
		if split_feature_value < tree['splitting_feature'][1]:
			return make_prediction(tree['left'], x, annotate)
		else:
			return make_prediction(tree['right'], x, annotate)	
def output(tree, test):
	prediction = [make_prediction(tree, x) for x in test]
	return prediction

if __name__ == '__main__':
	#read in data, label
	data = pd.read_excel("mlr06.xls")
	test = [[478, 184, 40, 74, 11, 31], [1000,10000,10000,10000,10000,1000,100000]]
	label = data['X7']
	del data['X7']
	node_id = 0
	global node_id
	
	all_pos_split = {}
	for col in range(data.shape[1]):
		all_pos_split[col] = get_splitting_points(data.iloc[:,col].tolist())

	tree = create_tree(data, all_pos_split, label, current_depth = 0)
	print output(tree, test)
