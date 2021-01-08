# regression tree
# input is a dataframe of features
# the corresponding y value(called labels here) is the scores for each document

from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
import scipy.optimize

node_id = 0


def get_splitting_points(args):
    # given a list
    # return a list of possible splitting values
    attribute, col = args
    attribute.sort()
    possible_split = []
    for i in range(len(attribute) - 1):
        if attribute[i] != attribute[i + 1]:
            possible_split.append(np.mean((attribute[i], attribute[i + 1])))
    return possible_split, col


# create a dictionary, key is the attribute number, value is whole list of possible splits for that column
def find_best_split_parallel(args):
    best_ls = 1000000
    best_split = None
    best_children = None
    split_point, data, label = args
    key, possible_split = split_point

    for split in possible_split:
        children = split_children(data, label, key, split)

        # weighted average of left and right ls
        ls = len(children[1]) * least_square(children[1]) / len(label) + len(children[3]) * least_square(
            children[3]) / len(label)
        if ls < best_ls:
            best_ls = ls
            best_split = (key, split)
            best_children = children
    return best_ls, best_split, best_children


def find_best_split(data, label, split_points):
    # split_points is a dictionary of possible splitting values
    # return the best split
    best_ls = 1000000
    best_split = None
    best_children = None
    pool = Pool()
    for ls, split, children in pool.map(find_best_split_parallel,
                                        zip(split_points.items(), repeat(data), repeat(label))):
        if ls < best_ls:
            best_ls = ls
            best_split = split
            best_children = children
    pool.close()

    return best_split, best_children  # return a tuple(attribute, value)


def split_children(data, label, key, split):
    left_index = [index for index in range(len(data.iloc[:, key])) if data.iloc[index, key] < split]
    right_index = [index for index in range(len(data.iloc[:, key])) if data.iloc[index, key] >= split]
    left_data = data.iloc[left_index, :]
    right_data = data.iloc[right_index, :]
    left_label = [label[i] for i in left_index]
    right_label = [label[i] for i in right_index]

    return left_data, left_label, right_data, right_label


def least_square(label: list):
    if not len(label):
        return 0
    return (np.sum(label) ** 2) / len(set(label))


def create_leaf(label: list):
    global node_id
    node_id += 1
    leaf = {
        'splittng_feature': None, 'left': None, 'right': None, 'is_leaf': True, 'index': node_id,
        'value': round(np.mean(label), 3)
    }
    return leaf


def find_splits_parallel(args: tuple):
    var_space, label, col = args
    # var_space = data.iloc[:,col].tolist()
    return scipy.optimize.fminbound(
        error_function, min(var_space), max(var_space), args=(col, var_space, label),
        full_output=True
    )


# return,
# if not min_error or error < min_error:
# 	min_error = error
# 	split_var = col
# 	min_split = split

def create_tree(data, all_pos_split, label, max_depth, ideal_ls, current_depth=0):
    remaining_features = all_pos_split
    # stopping conditions
    if sum([len(v) != 0 for v in remaining_features.values()]) == 0:
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(label)
    # #Additional stopping condition (limit tree depth)
    elif current_depth > max_depth:
        return create_leaf(label)

    #######
    min_error = None
    split_var = None
    min_split = None

    var_spaces = [data.iloc[:, col].tolist() for col in range(data.shape[1])]
    cols = [col for col in range(data.shape[1])]
    pool = Pool()
    for col, split, error, ierr, numf in enumerate(
            pool.map(find_splits_parallel, zip(var_spaces, repeat(label), cols))):
        if not min_error or error < min_error:
            min_error = error
            split_var = col
            min_split = split
    pool.close()

    splitting_feature = (split_var, min_split)
    children = split_children(data, label, split_var, min_split)

    left_data, left_label, right_data, right_label = children
    if len(left_label) == 0 or len(right_label) == 0:
        return create_leaf(label)

    left_least_square = least_square(left_label)

    # Create a leaf node if the split is "perfect"
    if left_least_square < ideal_ls:
        return create_leaf(left_label)
    if least_square(right_label) < ideal_ls:
        return create_leaf(right_label)

    # recurse on children
    left_tree = create_tree(left_data, remaining_features, left_label, max_depth, ideal_ls, current_depth + 1)
    right_tree = create_tree(right_data, remaining_features, right_label, max_depth, ideal_ls, current_depth + 1)
    return {
        'is_leaf': False,
        'value': None,
        'splitting_feature': splitting_feature,
        'left': left_tree,
        'right': right_tree,
        'index': None
    }


def error_function(split_point, split_var, data, label):
    data1 = []
    data2 = []
    for i in range(len(data)):
        temp_dat = data[i]
        if temp_dat <= split_point:
            data1.append(label[i])
        else:
            data2.append(label[i])
    return least_square(data1) + least_square(data2)


def make_prediction(tree, x, annotate=False):
    if tree['is_leaf']:
        if annotate:
            print("At leaf, predicting %s" % tree['value'])
        return tree['value']
    else:
        # the splitting value of x.
        split_feature_value = x[tree['splitting_feature'][0]]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value < tree['splitting_feature'][1]:
            return make_prediction(tree['left'], x, annotate)
        else:
            return make_prediction(tree['right'], x, annotate)


class RegressionTree:
    def __init__(self, training_data, labels, max_depth=5, ideal_ls=100):
        self.training_data = training_data
        self.labels = labels
        self.max_depth = max_depth
        self.ideal_ls = ideal_ls
        self.tree = None

    def fit(self):
        global node_id
        node_id = 0
        all_pos_split = {}
        pool = Pool()
        splitting_data = [self.training_data.iloc[:, col].tolist() for col in range(self.training_data.shape[1])]
        cols = [col for col in range(self.training_data.shape[1])]
        for dat, col in pool.map(get_splitting_points, zip(splitting_data, cols)):
            all_pos_split[col] = dat
        pool.close()
        self.tree = create_tree(self.training_data, all_pos_split, self.labels, self.max_depth, self.ideal_ls)

    def predict(self, test):
        prediction = np.array([make_prediction(self.tree, x) for x in test])
        return prediction


if __name__ == '__main__':
    # read in data, label
    data = pd.read_excel("mlr06.xls")
    test = [[478, 184, 40, 74, 11, 31], [1000, 10000, 10000, 10000, 10000, 1000, 100000]]
    label = data['X7']
    del data['X7']

    model = RegressionTree(data, label)
    model.fit()
    print(model.predict(test))
