import numpy as np
import scipy

import sys
sys.path.append("..")
from utils.models.DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, 
                 n_trees = 10,
                 criterion = "gini", 
                 max_depth = None, 
                 min_samples_split = 2, 
                 min_samples_leaf = 1, 
                 min_impurity_decrease = 0.0,
                 max_thresholds = 1,
                 random_state = 0):
        
        """
            Initialize the decision tree parameters.
            
            Args:
                - Parameters of RF:
                    n_trees (int): number of trees to be built
                    random_state (int): random seed.
                
                - Paramters of DT to be built: 
                    criterion (str): the function to measure the quality of a split ("gini" or "entropy").
                    max_depth (int): the maximum depth of the tree.
                    min_samples_split (int): the minimum number of samples required to split an internal node.
                    min_samples_leaf (int): the minimum number of samples required to be at a leaf node.
                    min_impurity_decrease (float): the minimum impurity decrease required to split a node.
                    max_thresholds (int): the maximum number of thresholds to use during best split search.
                    max_features (str / float): the maximum number of features to choose from in the given ones: 
                                                    - if "sqrt", then int(sqrt(n_features)); 
                                                    - if "log2", then int(log2(n_features)):
                                                    - if float, then int(n_features * max_features);
                                                    - if None, all features
                    random_state (int): random seed.
        """
        
        self.n_trees = n_trees

        #When the traing process is complete, all the DecisionTree objects in this list will be trained
        self.trees = [DecisionTree(criterion = criterion,
                                   max_depth = max_depth,
                                   min_samples_split = min_samples_split,
                                   min_samples_leaf = min_samples_leaf,
                                   min_impurity_decrease = min_impurity_decrease,
                                   max_thresholds = max_thresholds,
                                   max_features = "sqrt", #As explained in the theory part, for classification tasks, sqrt(n_features) have to be used.
                                                          #Thus, we pass directly to the DT the parameter max_features = "sqrt", so that it will be the DT to selct randomly them;
                                                          #see for more details methods "build_tree(...)" and "find_best_split(...)" of DT.
                                   random_state = random_state)
                     for _ in range(self.n_trees)]

        self.random_state = random_state

    def fit(self, X, y):
        """
            Fit a RF model with the specified parameters to the data. 

            Args:
                X (np.ndarray): the matrix containing the features (sample on the rows);
                y (np.ndarray): the vector containg the labels.
        """
 
        for i in range(self.n_trees):
            #For each subtree that we have to build, we generate a bootstrapped dataset
            bs_ids = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace = True)
            X_bs = X[bs_ids]
            y_bs = y[bs_ids]

            #Then, we train the DT on it
            self.trees[i].fit(X_bs, y_bs)

    def predict(self, X):
        """
            This method predicts the labels of a given set of samples.

            Args:
                X (np.ndarray): the matrix containing the samples to be classified (samples on rows)

            Returns:
                labels (np.ndarray): the predicted label of each sample in X
                scores (np.ndarray): the score associated to each prediction
        """

        #As define in the theory part, we collect the prediction of each tree.
        #Note: the resulting matrix is n_trees x n_samples, so we have to compute the frequencies of the values
        #on the columns, not on the rows!
        predictions = np.array([t.predict(X) for t in self.trees])  
        
        #scipy.stats.mode (axis = 0) collets all the modes or "modal" (most common values) along with their frequencies
        labels, counts = scipy.stats.mode(predictions, axis = 0)

        #Finally, we normalize the frequencies so that tey are in [0,1]
        scores = counts / predictions.shape[0]

        return labels, scores