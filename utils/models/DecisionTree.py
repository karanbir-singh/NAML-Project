import numpy as np
from utils.models.base_model import BaseModel

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None, class_prob = None):
        """
            Initializes the node.
            Args:
                feature (int): the feature that the node is splitting on.
                threshold (float): the threshold that the node is splitting on.
                left (Node): the left child of the node.
                right (Node): the right child of the node.
                value (float): the value of the node if it is a leaf node.
        """
        
        self.feature = feature
        
        self.threshold = threshold

        self.left = left
        self.right = right

        self.value = value

        self.class_prob = class_prob

class DecisionTree(BaseModel):
    def __init__(self, 
                 criterion = "gini", 
                 max_depth = None, 
                 min_samples_split = 2, 
                 min_samples_leaf = 1, 
                 min_impurity_decrease = 0.0,
                 max_thresholds = 1,
                 max_features = None,
                 random_state = 0):
        """
            Initialize the decision tree parameters.
            
            Args:
                criterion (str): the function to measure the quality of a split ("gini" or "entropy").
                max_depth (int): the maximum depth of the tree.
                min_samples_split (int): the minimum number of samples required to split an internal node.
                min_samples_leaf (int): the minimum number of samples required to be at a leaf node.
                min_impurity_decrease (float): the minimum impurity decrease required to split a node.
                max_thresholds (int): the maximum number of thresholds to use during best split search.
                max_features (str / float): the maximum number of features to choose from when determining the best splits
                                            (note: they change at each call of "best_split"): 
                                                - if "sqrt", then int(sqrt(n_features)); 
                                                - if "log2", then int(log2(n_features)):
                                                - if float, then int(n_features * max_features);
                                                - if int, then min(n_features, max_features);
                                                - if None, all features
                random_state (int): random seed.
        """

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_thresholds = max_thresholds
        self.max_features = max_features
        self.random_state = random_state

        #This attribute is not defined in the constructor because it's known only when y data is passed (self.fit(...))
        #It will be used to make the length of the arrays of probabilities all equal to num_classes
        self.num_classes = None

        self.tree = None

        np.random.seed(random_state)

    def entropy(self, y):
        """
            Calculate entropy for a target variable y.

            Args:
                y (np.ndarray): the target variable.

            Returns:
                entropy (float): the entropy of the target variable.
        """

        #Compute the proportions of each class in the target variable y
        proportions = np.bincount(y) / len(y)

        #To avoid log(0), we consider only the proportions greater than 0
        gt_zero = proportions > 0

        return -np.sum(proportions[gt_zero] * np.log2(proportions[gt_zero]))

    def gini(self, y):
        """
            Calculate Gini impurity for a target variable y.
            
            Args:
                y (np.ndarray): the target variable.

            Returns:
                gini (float): the Gini impurity of the target variable.
        """

        #Compute the proportions of each class in the target variable y
        proportions = np.bincount(y) / len(y)

        return 1 - np.sum(proportions ** 2)
    
    def split_samples(self, X, y, feature_idx, threshold, mask = None):
        """
            Utility function to split the dataset based on a feature index and threshold.
            Samples with feature values less than or equal to the threshold go to the left child, the rest go to the right child.

            Args:
                X (np.ndarray): the feature matrix.
                y (np.ndarray): the target variable.
                feature_idx (int): the index of the feature to split on.
                threshold (float): the threshold to split the feature on.
                mask (list): list of booleans indicating which parameters to return. Order: [X_left, X_right, y_left, y_right]; if None, all parameters are returned.

            Returns:
                tuple: contains only the requested parameters based on the mask. Full order is (X_left, X_right, y_left, y_right).
        """

        #As defined in the theory part, the samples with feature values less than or equal to the threshold go to the left child, the rest go to the right child
        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx

        #If the mask is None, all values are returned
        if mask is None:
            return X[left_idx], X[right_idx], y[left_idx], y[right_idx]
        
        #Otherwise, choose only the ones with mask == True
        results = []
        if mask[0]: results.append(X[left_idx])
        if mask[1]: results.append(X[right_idx])
        if mask[2]: results.append(y[left_idx])
        if mask[3]: results.append(y[right_idx])
        
        return tuple(results)

    def find_best_split(self, X, y):
        """
            Find the best feature and threshold to split on.

            Args:
                X (np.ndarray): the feature matrix.
                y (np.ndarray): the target variable.

            Returns:
                split_idx (int): the index of the feature to split on, None if no split has been found.
                split_threshold (float): the threshold to split the feature on, None if no split has been found.
                best_gain (float): the information gain of the best split, -1.0 if no split has been found.
        """

        #Initialize the best gain to -1.0 (minimum value, at least it should be 0)
        best_gain = -1.0

        #Initialize the split index and split threshold to None (no split considered yet)
        split_idx, split_threshold = None, None
        
        features = range(X.shape[1])

        #If "max_features" is a int, then min(max_features, X.shape[1]) features will be evaulated
        if type(self.max_features) == int:
            features = np.random.choice(features, size = min(self.max_features, X.shape[1]), replace = False)
        
        #If "max_features" is a float, then int(max_features * num_features) features will be evaulated
        if type(self.max_features) == float:
            features = np.random.choice(features, size = int(X.shape[1] * self.max_features), replace = False)
        
        #If "max_features" is "sqrt", then int(sqrt(n_features)) features will be evaulated
        if self.max_features == "sqrt":
            features = np.random.choice(features, size = int(np.sqrt(X.shape[1])), replace = False)
        
        #If "max_features" is "log2", then int(log2(n_features)) features will be evaulated
        if self.max_features == "log2":
            features = np.random.choice(features, size = int(np.log2(X.shape[1])), replace = False)

        #If all of this conditions fail (i.e. "self.max_features" is None or another unacceptable value), "features" will not be still range(X.sahpe[1])

        for feature_idx in features:
            #Remove the duplictes from the feature values, sort it and compute the pairwise mean
            values = np.sort(np.unique(X[:, feature_idx]))
            pairwise_mean = (values[:-1] + values[1:]) / 2

            #Thresholds are chosen randomly between all possible values of the pairwise means
            #Another idea is to choose them dicrectly on "values" (avoid sort and pairwise mean, but not unique)
            thresholds = np.random.choice(pairwise_mean, size = min(len(np.unique(pairwise_mean)), self.max_thresholds), replace = False)

            #For each threshold, calculate the information gain and choose the best threshold
            for threshold in thresholds:
                #Split the dataset (X, y) based on the feature and threshold
                #Same as the split function, samples with feature values less than or equal to the threshold go to the left child, the rest go to the right child
                y_left, y_right = self.split_samples(X, y, feature_idx, threshold, [False, False, True, True])

                #The split is considered to be invalid if the number of samples in the left or right child is less than the minimum number of samples required to split
                #In this case, we skip to the next threshold
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                #Compute impurity for the split
                gain = self.compute_information_gain(y, y_left, y_right, self.gini if self.criterion == "gini" else self.entropy)
                
                #If the information gain is greater than the best gain, update the best gain, split index and split threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold, best_gain

    def compute_information_gain(self, y, y_left, y_right, impurity):
        """
            This method is used to compute inromation gain.

            Args:
                y (np.ndarray): the target variable.
                y_left (np.ndarray): the target variable in the left child.
                y_right (np.ndarray): the target variable in the right child.
                impurity (function): the impurity function to use.

            Returns:
                information_gain (float): the information gain.
        """
        
        #First, calculate the impurity of the parent node
        parent_loss = impurity(y)

        #The, compute the number of samples in the parent, left child and right child
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        #Finally, calculate the impurity of the left and right children and compute the information gain with the formula defined in the theory part
        child_loss = (1.0 * n_left * impurity(y_left) + 1.0 * n_right * impurity(y_right)) / n

        return parent_loss - child_loss

    def build_tree(self, X, y, depth = 0):
        """
            This method recursively build the decision tree by applying the algorithm described in the theory part.

            Args:
                X (np.ndarray): the feature matrix.
                y (np.ndarray): the target variable.
                depth (int): the current depth of the tree.
            
            Returns:
                node (Node): the root node of the decision tree.
        """

        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        #Three stopping criteria are considered:
        #1. If the maximum depth is reached
        #2. If there is only one label in the target variable
        #3. If the number of samples is less than the minimum number of samples required to split
        #If at least one condition is True, return a leaf node with the most common label
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            class_counts = np.bincount(y, minlength = self.num_classes)
            return Node(value = np.argmax(class_counts), class_prob = class_counts / np.sum(class_counts))

        #Find the best split (i.e., the best feature and threshold to split on)
        feature_idx, threshold, best_gain = self.find_best_split(X, y)
        
        #If the best split is invalid (i.e., feature_idx is None or best_gain is less than the minimum impurity decrease), return a leaf node with the most common label.
        if feature_idx is None or best_gain < self.min_impurity_decrease:
            class_counts = np.bincount(y, minlength = self.num_classes)
            return Node(value = np.argmax(class_counts), class_prob = class_counts / np.sum(class_counts))

        #Split the dataset based on the best feature and threshold
        X_left, X_right, y_left, y_right = self.split_samples(X, y, feature_idx, threshold)

        #This is the recursive part of the algorithm: we are using DFS, starting from the root node and going down to the leaf nodes along the branches.
        #The left child is built by calling the build_tree method with the left child dataset (X_left, y_left) and the depth increased by 1; same for the right child.
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        #At this point we have built the left and right children of the current node.
        #If both children are leaves and they have same most common label, consider deleting the leafs and assigning the parent node the most common label making it a leaf node
        #if left_child.value is not None and right_child.value is not None and left_child.value == right_child.value:
        #    return Node(value = left_child.value, class_prob = (len(y_left) * left_child.class_prob + len(y_right) * right_child.class_prob) / (len(y_left) + len(y_right)))
        #else:
        return Node(feature = feature_idx, threshold = threshold, left = left_child, right = right_child, class_prob = None)

    def fit(self, X, y):
        """
            This method is the entry point to build the decision tree from the training data: without it we woudn't be able to initialize self.tree.
            Important:  it is necessary to call this method before making predictions.

            Args:
                X (np.ndarray): the feature matrix.
                y (np.ndarray): the target variable.
        """

        self.num_classes = len(np.unique(y))
        self.tree = self.build_tree(X, y)

    def traverse_tree(self, x, node):
        """
            This is a utility method used to traverse the tree to make a prediction.
            
            Args:
                x (np.ndarray): the feature vector.
                node (Node): the current node.
            
            Returns:
                prediction (int): the predicted class.
        """

        #If the node is a leaf node, return the value of the node
        if node.value is not None:
            return node

        #If the feature value of the sample x is less than or equal to the threshold of the node, go to the left child, otherwise go to the right child
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X):
        """
            Predict the class for each sample in X.

            Args:
                X (np.ndarray): the feature matrix.
            
            Returns:
                predictions (np.ndarray): the predicted classes.
        """

        #Get the leaf the ample belongs to
        leafs = [self.traverse_tree(x, self.tree) for x in X]

        #For each sample in X, we traverse the tree to make a prediction
        return np.array([l.value for l in leafs]), np.array([l.class_prob for l in leafs])