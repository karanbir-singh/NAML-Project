import numpy as np
import matplotlib.pyplot as plt
from utils.models.base_model import BaseModel

class GaussianNaiveBayes(BaseModel):
    def __init__(self, classes):
        """
            Initializes the Naive Bayes classifier parameters.

            Args:
                classes (list): the classes the data is classified in.
                                Note: this parameter can be removed and its usages replaced with [0, 1], but since
                                NB can be used for multi-class problems, it seems reasonable to us keeping it. 
        """

        self.classes = classes

        #A dict that will contain the values of the priors indexed by class.
        self.priors = None

        #Each value will be an array of len(features) (the columns of X passed to self.fit())
        #containing the mean and the variance of Gaussian distribution associated to (y_k, X_i) as defined in the theory part
        self.means = None
        self.variances = None
    
    def fit(self, X, y): 
        """
            This methods fits NB to the training data.

            Args:
                X (np.ndarray): the matrix containing the training samples (on the rows).
                y (np.ndarray): the vector with the labels.
        """

        #Compute and store the priors (see theory part)
        self.priors = {}

        for c in self.classes:
            self.priors[c] = np.mean(y == c)
        
        #Compute and store mean and variance for each feature per class (see theory part)
        self.means = {}
        self.variances = {}

        max_var = np.var(X, axis = 0).max()
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis = 0)

            #Note: here we could have used the NumPy built-in command for the variance estimation:
            #np.var(X_c, axis = 0, ddof = 1) where ddof = 1 means that at the denominator we want len(X_c) - 1
            #to get an unbiased estimator.
            #Note: 1e-9 * max_var is a smoothing parameter useful when variances have different scales (like Time and the Vs)
            self.variances[c] = np.sum((X_c - self.means[c]) ** 2, axis = 0) / (len(X_c) - 1) + 1e-9 * max_var

    def predict(self, X):
        """
            This method makes predictions on new data.

            Args:
                X (np.ndarray): the matrix containing the samples to predict (on the rows).

            Returns:
                predictions (np.ndarray): the array containing the predicted class.
                probabilities (np.ndarray): the probability associated to the prediction
        """

        if self.means == None or self.variances == None:
            raise ValueError("NB classifier has not been trained yet.")
        
        #Posterior is given by P{Y = y_k | X}, so we represent it in a matrix form where each row corresponds to a sample and 
        #each column corresponds the probability of a class y_k. 
        #Note: in the case of binary classification, we can reduce this matrix to a vector. For the sake of generality, we keep this form. 
        posteriors = np.empty((X.shape[0], len(self.classes)))
        
        for i, c in enumerate(self.classes):
            #Both priors and likelihoods can be really small numbers, so we'll work using the log 
            prior = np.log(self.priors[c])

            #For the likelihood, we have to compute log(*) = log(density(X, self.means[c], self.variances[c])) where
            #"density" is the Gaussian PDF, so by log(ab) = log(a) * log(b):
            #log(*) = log(1 / np.sqrt(2 * np.pi * self.variances[c] + 1e-8)) * log(np.exp((X - self.means[c]) ** 2 / (2 * self.variances[c] + 1e-8))) =
            #       = -1 / 2 * np.log(2 * np.pi * self.variances[c] + 1e-8) - (X - self.means[c]) ** 2 / (2 * self.variances[c] + 1e-8)
            likelihood = np.sum(-0.5 * np.log(2 * np.pi * self.variances[c] + 1e-8) - (X - self.means[c]) ** 2 / (2 * self.variances[c] + 1e-8), axis = 1)

            #Finally, we store the log of the posterior in the matrix. 
            #Note: we are saving one entire column at the time because we'are working column by column. "prior" is a single number, but "likelihood" is a vector
            #corresponding to the density evaluated in the sample
            posteriors[:, i] = prior + likelihood
        
        #Since probabilities can be very small (because we're working with log the numbers will be far negative), we use log-sum-exp trick for numerical stability
        #First, find the maximum value for each row: this will be then converted to 1 when we'll take the exp
        max_posteriors = np.max(posteriors, axis = 1, keepdims = True)
        
        #Then, subtract the maximum from each value and exp: all the resulting numbers will be less then (or equal to) 1 because posteriors - max_posteriors <= 0
        #To get the real probabilities, this numbers need to be rescaled by e^(-max_posteriors)
        exp_posteriors = np.exp(posteriors - max_posteriors)
        
        #Finally, normalize to get probabilities.
        #What we are computing here is e^(posteriors - max_posteriors) / sum(e^(posteriors - max_posteriors))) = e^(posteriors) / sum(e^(posteriors)) that is the 
        #probability we're interested in, but without risking underflow!
        probs = exp_posteriors / np.sum(exp_posteriors, axis = 1, keepdims = True)

        #The predicted class is the one with highest probability (see theory)
        return self.classes[np.argmax(probs, axis = 1)], probs[:, 1]
    
    def plot_gaussian_pdfs(self, feature_names, n_cols = 1):
        """
            Utility method to plot the probability density function associated to each class and features.
            PDFs of same feature will be displayed on the same plot.

            Args:
                feature_name (list): the names to be assigned to the features. This list should contain only
                                     the features used for training in that order.
                n_cols (tuple): specifies how many plots to be displayed on each row
        """

        n_rows = int(np.ceil(len(feature_names) / n_cols)) 

        fig, axs = plt.subplots(n_rows, n_cols, figsize = (5 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i in range(len(feature_names)):
            for k in range(len(self.classes)):
                #This is a definition used only to avoid to repeat in the code a lot of times self.classes[k] given that the dicts are indexed by class
                c = self.classes[k]

                #Use a sufficiently large range for the x variable and compute the Gaussian PDF associated to self.means[c] and self.variances[c]
                x = np.linspace(self.means[c][i] - 5 * self.variances[c][i], self.means[c][i] + 5 * self.variances[c][i], 250)
                y = 1.0 / np.sqrt(2 * np.pi * self.variances[c][i]) * np.exp(-((x - self.means[c][i]) ** 2) / (2 * self.variances[c][i]))

                #Finally, plot all PDFs for same feature on the same plot
                #Note: there might be classes with PDF with small variance on the same plot as others with higher one!
                axs[i].plot(x, y, "--", label = f"class: {self.classes[k]}")
                axs[i].set_title(f"Feature name: {feature_names[i]}")
                axs[i].set_xlabel("x")
                axs[i].set_ylabel("PDF")
            
            axs[i].legend()

        #Remove empty subplots if any
        for i in range(len(feature_names), n_rows * n_cols):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.plot()