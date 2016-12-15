# LambdaMart
Python implementation of LambdaMart

LambdaMART API:
LambdaMART(training_data=None, number_of_trees=0, leaves_per_tree=0, learning_rate=0)

Parameters:

	training_data: Numpy array of documents (default: None)
		Each document’s  format is [relevance score, query index, feature vector]
	number_of_trees: int (default: 5)
		Number of trees LambdaMART goes through
	learning_rate: float (default: 0.1)
		Rate at which we update our prediction with each tree
	tree_type: string (default: “sklearn”)
		Either “sklearn” for using Sklearn implementation of the tree or “original” for 
		using our implementation of the tree.


Methods:

	fit: Fits the model on the training data.
		Parameters: None
		Returns: None
	predict: Predicts the scores for the test dataset.
		Parameters: Numpy array of documents with each document’s format is [query index, feature vector] 
		Returns: Numpy array of scores
	validate: Predicts the scores for the test dataset and calculates the NDCG value.
		Parameters: Numpy array of documents with each record’s format is [relevance score, query index, feature vector] 
		Returns: NDCG value and Numpy array of scores
	save: Saves the model into a “.lmart” file with the name given as a parameter
		Parameters: Filename
		Returns: None
	load: Loads the model from the “.lmart” file given as a parameter
		Parameters: Filename
		Returns: None


#Tutorial:

To start using the API, you need to include the files: “lambdamart.py” and “RegressionTree.py” in the same directory.
Create a Python file in the same directory as the other Python files and for the sake of this tutorial, call it “example.py”. To run this example, you will need a training dataset and a test dataset. You can download the training dataset here https://github.com/lezzago/LambdaMart/blob/master/example_data/train.txt: and the test dataset here: https://github.com/lezzago/LambdaMart/blob/master/example_data/test.txt.

#Step 1: Import needed packages
In the “example.py” file, you will need to import lambdamart and numpy to pass in the data in the correct format like below:
from lambdamart import LambdaMART
import numpy as np

#Step 2: Create a function to pass in the data properly from the given training and test datasets.
```python
def get_data(file_loc):
	f = open(file_loc, 'r')
	data = []
	for line in f:
		new_arr = []
		arr = line.split(' #')[0].split()
		''' Get the score and query id '''
		score = arr[0]
		q_id = arr[1].split(':')[1]
		new_arr.append(int(score))
		new_arr.append(int(q_id))
		arr = arr[2:]
		''' Extract each feature from the feature vector '''
		for el in arr:
			new_arr.append(float(el.split(':')[1]))
		data.append(new_arr)
		f.close()
		return np.array(data)
```

#Step 3: Call the get_data function to get the training and test datasets. Also put it in a main function.
```python
def main():
	training_data = get_data(<Location to training file>)
	test_data = get_data(<Location to test file>)
```
Please replace the “<>” and the contents in them with the appropriate file locations.


#Step 4: Call LambdaMART, fit the data, and put it under the main function.
```python
model = LambdaMART(
	training_data=training_data, 
	number_of_trees=2, 
	learning_rate=0.1)
model.fit()
```
Please note that you can set the parameters to your specifications. Also note that the higher number of trees will make the program slower.


#Step 5: Run prediction or validation and put it under the main function
```python
average_ndcg, predicted_scores = model.validate(test_data)
predicted_scores = model.predict(test_data[:,1:])
```
Please note that predicted_scores from predict and validate are the same. Also the predict function cannot contain the relevance score, so that column needs to be omitted like is has been done above.


Now you have a working example for running the LambdaMART algorithm.
If you want to save this model after training it, please put this line below “model.fit()”:
```python
model.save(‘example_model’)
```
This will save a file called “example_model.lmart”.
To load this file, replace Step 4 with this:
```python
model = LambdaMART()
model.load(‘example_model.lmart’)
```
This will create a new LambdaMART object and load the model from the “example_model.lmart” file.
