import numpy as np
import RegressionTree

#lambda,weight,assign:np.array


def group_by_node(assign):

	# assign: a vector indicates the node index in the tree
	# for each sample.

	nodes = list(set(assign))
	groups = []
	for i in nodes:
		group = []
		for j in range(len(assign)):
			if i == assign[j]:
				group.append(j)
		groups.append(group)

	return groups

def essemble_trees(lmda,w,assign,ori_score,lr):

	#ori_score: original score for each sample
	#lr: learning rate

	#This function output the updated score for 
	#each sample.

	groups = group_by_node(assign)
	gamma_vec = []
	for group in groups:
		lmda_group = [lmda[i] for i in group]
		w_group = [w[i] for i in group]

		gamma = float(sum(lmda_group))/sum(w_group)
		gamma_vec.append(gamma)

	update_vec = np.array([gamma_vec[ele-1] for ele in assign])

	score = ori_score + lr*update_vec

	return score


	
'''
test:
essemble_trees([1,2,3,4,5,6,7],[7,6,5,4,3,2,1],[1,1,1,1,2,2,2],[0,0,0,0,0,0,0],0.1)
'''
