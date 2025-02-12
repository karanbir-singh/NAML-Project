import numpy as np
import scipy

import sys
sys.path.append("..")
from utils.models.DecisionTree import DecisionTree

from utils.models.base_model import BaseModel

class RandomForest(BaseModel):
    def __init__(self, 
                 n_trees = 10,
                 criterion = "gini", 
                 max_depth = None, 
                 min_samples_split = 2, 
                 min_samples_leaf = 1, 
                 min_impurity_decrease = 0.0,
                 max_thresholds = 1,
                 max_features = "sqrt", #As explained in the theory part, for classification tasks, sqrt(n_features) is typically used.
                                        #We choose to not pass directly this value to the DT because in some contexts, user might want to change it;
                                        #see for more details methods "build_tree(...)" and "find_best_split(...)" of DT.
                 random_state = 0):
        
        """
            Initializes both random forest' and decision tree' parameters.
            
            Args:
                - Parameters of RF:
                    n_trees (int): number of trees to be built
                    random_state (int): random seed.
                
                - Parameters of DT to be built: 
                    criterion (str): the function to measure the quality of a split ("gini" or "entropy").
                    max_depth (int): the maximum depth of the tree.
                    min_samples_split (int): the minimum number of samples required to split an internal node.
                    min_samples_leaf (int): the minimum number of samples required to be at a leaf node.
                    min_impurity_decrease (float): the minimum impurity decrease required to split a node.
                    max_thresholds (int): the maximum number of thresholds to use during best split search.
                    max_features (str / float): the maximum number of features to choose from in the given ones: 
                                                    - if "sqrt", then `int(sqrt(n_features))`;
                                                    - if "log2", then `int(log2(n_features))`;
                                                    - if float, then `int(n_features * max_features)`;
                                                    - if int, then `min(n_features, max_features)`;
                                                    - if None, all features
                    random_state (int): random seed.
        """
        
        self.n_trees = n_trees

        #When the training process is complete, all the DecisionTree objects in this list will be trained
        self.trees = [DecisionTree(criterion = criterion,
                                   max_depth = max_depth,
                                   min_samples_split = min_samples_split,
                                   min_samples_leaf = min_samples_leaf,
                                   min_impurity_decrease = min_impurity_decrease,
                                   max_thresholds = max_thresholds,
                                   max_features = max_features,
                                   random_state = random_state)
                     for _ in range(self.n_trees)]

        self.random_state = random_state

    def fit(self, X, y):
        """
            Fit a RF model with the specified parameters to the data. 

            Args:
                X (np.ndarray): the matrix containing the features (sample on the rows);
                y (np.ndarray): the vector containing the labels.
        """
 
        for i in range(self.n_trees):
            #For each subtree that we have to build, we generate a bootstrapped dataset
            bs_ids = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace = True)
            X_bs = X[bs_ids]
            y_bs = y[bs_ids]

            #Then, we train the DT on it
            self.trees[i].fit(X_bs, y_bs)

    def predict(self, X, type = "prob"):
        """
            This method predicts the labels of a given set of samples.

            Args:
                X (np.ndarray): the matrix containing the samples to be classified (samples on rows).
                type (str): if "prob", predicted labels and average probabilities for them are returned,
                            if "voting", predicted labels and voting fraction are returned.
 
            Returns:
                labels (np.ndarray): the predicted label of each sample in `X`.
                scores (np.ndarray): the score associated to each prediction (see "type" attribute).
        """

        #As defined in the theory part, we collect the predictions and probabilities of each tree.
        pred_labels, pred_probs = zip(*[t.predict(X) for t in self.trees])

        #Note: this will a matrix of shape n_trees x n_samples because the prediction is a number 
        pred_labels = np.array(pred_labels)

        #Note: this will be a tensor of shape n_trees x n_samples x n_classes because each tree is outputting a vector of probabilities
        pred_probs = np.array(pred_probs)

        if type == "prob":          
            #First, compute the average probability across all trees: we are working row by row (each row is n_samples x n_classes),
            #taking the average of each array of probabilities [... n_classes ...]. The resulting matrix will be n_samples x n_classes
            avg_probs = np.mean(pred_probs, axis = 0)
            
            #Each column of "avg_probs" contains the average probabilities for a specific class; thus, to have the label of each sample, we have to take the 
            #argmax over all the columns argmax([... n_classes ...]).
            labels = np.argmax(avg_probs, axis = 1)
    
            return labels, avg_probs
        elif type == "voting":
            #For each sample, voting fractions are represented by an array of length n_classes and we have n_samples samples, so
            #the matrix has shape (n_samples, n_classes)
            voting_fractions = np.zeros((X.shape[0], self.trees[0].num_classes))  

            #Each row of pred_labels contains the labels assigned to all the samples by a specific tree, so to compute the voting fraction, we need to 
            #count the frequencies of each label in a certain column and then normalize them
            for sample_idx in range(X.shape[0]):
            
                #Count the number of trees voting for each class for the current sample: min_length = num_classes is used so that even if all 
                #trees agree on a certain label the resulting voting_fractions array will have length n_classes ([0, 0, ..., 1, ..., 0, 0])
                class_counts = np.bincount(pred_labels[:, sample_idx], minlength = self.trees[0].num_classes)
                
                #Then, normalize to obtain a value in [0, 1]
                voting_fractions[sample_idx] = class_counts / self.n_trees

            #The predicted label is the one with the largest voting fraction
            labels = np.argmax(voting_fractions, axis = 1)

            return labels, voting_fractions
        else:
            raise ValueError(f"Parameter 'type' should be either 'prob' or 'voting', got {type}")