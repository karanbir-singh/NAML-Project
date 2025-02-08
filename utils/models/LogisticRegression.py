import jax
import jax.numpy as jnp

from utils.data_processer import *
from utils.models.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, num_features=None):
        """
        A class representing a logistic regression classifier configuration for specifying the number of features
        the classifier has to manage.

        :param num_features: The number of features of the dataset.
        :type num_features: Optional[int]
        """

        self.num_features = num_features

    def initialize_parameters(self, num_features):
        """
            Returns the parameters (weights and bias) for the logistic regression given the number of the dataset's features,
            initialized randomly.

            :param num_features: The number of features which represents also the number of weights to generate
            :type num_features: int
            :return: Parameters of the logistic regression classifier, specifically weights and biases.
        """

        self.num_features = num_features

        np.random.seed(0)  # For reproducibility

        # Parameters
        weights = np.random.randn(self.num_features, 1)
        bias = 0.

        return weights, bias

    def MSW(self, weights):
        """
        Computes the mean squared weights (MSW) of a logistic regression classifier's weights.

        This function calculates the average of the squared values of the weights present
        in the given parameters of a logistic regression classifier.

        :param weights: Weights of the logistic regression classifier
        :type weights: list
        :return: Mean of squared weights of the logistic regression classifier
        :rtype: float
        """

        # Calculate MSW
        partial_sum = 0.0
        n_weights = 0
        for W in weights:
            partial_sum = partial_sum + jnp.sum(W * W)
            n_weights = n_weights + W.size

        return partial_sum / n_weights

    # Loss function
    def cross_entropy(self):
        """
        Defines the cross-entropy loss function for classification problems with
        logistic output. This function calculates the negative log-likelihood
        of the predictions made by the model when compared with the true labels.
        It is commonly used as a loss function for binary classification tasks.

        :returns: A callable function that computes the cross-entropy loss
            for given input features, true labels, and model parameters. The
            callable expects four arguments: x (input features), y (true labels),
            weights and bias (model parameters).
        :rtype: Callable
        """
        def callable(x, y, weights, bias):
            y_pred = self.predict(x, weights, bias)
            return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
        return callable

    def regularized_loss(self, loss_function, penalization):
        """
        Computes a regularized loss function by combining the given loss function with
        a penalization term. The penalization term is scaled based on the dataset size
        and involves the Mean Squared Weight (MSW) for the provided weights.

        This method returns a callable function that takes `x`, `y`, `weights` and `bias` as inputs
        and computes the regularized loss by applying the given loss function and adding
        the scaled penalization term.

        :param loss_function: A callable function representing the loss function to be
            applied to the dataset. It should accept `x`, `y`, `weights` and `bias` as inputs.
        :param penalization: A scalar value representing the penalization term to be
            applied during regularization. This value is combined with the MSW.
        :return: Returns a callable function which computes the regularized loss as a
            combination of the loss function and the penalization term.
        """
        def callable(x, y, weights, bias):
            return loss_function(x, y, weights, bias) + penalization / (2 * x.shape[0]) * self.MSW(weights)

        return callable

    # Optimisation algorithms
    def SGD(
            self,
            loss_function,
            epochs=1000,
            batch_size=128,
            learning_rate_min=1e-3,
            learning_rate_max=1e-1,
            learning_rate_decay=1000,
    ):
        """
        Creates a Stochastic Gradient Descent (SGD) optimizer as a callable function.

        This optimizer performs parameter updates based on gradient descent using
        a randomly selected batch of data at each iteration. The learning rate
        decays linearly over the specified number of epochs. The optimization
        process history, including the loss at each epoch, is also recorded and
        returned.

        :param loss_function: The loss function used to guide the optimization
            process. Should accept inputs, labels, and model parameters, and
            return a scalar loss value.
        :param epochs: The total number of iterations for training the model.
        :param batch_size: The number of samples to randomly select for computing
            gradients at each iteration.
        :param learning_rate_min: The minimum boundary for the learning rate.
        :param learning_rate_max: The initial maximum learning rate.
        :param learning_rate_decay: The rate at which the learning rate decreases
            linearly over the epochs. The learning rate reaches `learning_rate_min`
            when `epochs` are completed.
        :return: A callable optimizer function that applies SGD on the provided
            data and model parameters and returns the updated parameters and a
            history of loss values over all epochs.
        """
        def callable(x_train, y_train, weights, bias):
            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=[2,3]))

            # History
            history = list()
            history.append(loss(x_train, y_train, weights, bias))

            for epoch in range(epochs):
                # Get learning rate
                learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch/learning_rate_decay))

                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient
                grad_vals = grad_loss(x_train[idxs,:], y_train[idxs,:], weights, bias)

                # Update weights and bias
                weights = weights - learning_rate * grad_vals[0]
                bias = bias - learning_rate * grad_vals[1]

                # Update history
                history.append(loss(x_train, y_train, weights, bias))
            return weights, bias, history
        return callable

    def RMSprop(
            self,
            loss_function,
            epochs=1000,
            batch_size=128,
            learning_rate=0.1,
            decay_rate = 0.9,
            epsilon = 1e-8
    ):
        """
        Implements the RMSprop optimization algorithm. RMSprop is an adaptive learning rate method
        that maintains a moving average of the square of the gradients to normalize the gradient step sizes.
        This function returns a callable object capable of training a model using inputs, outputs, and
        initial parameters.

        :param loss_function: The loss function to be minimized. This function must take the training inputs,
            expected outputs, and model parameters as arguments and return the corresponding loss.
        :type loss_function: callable
        :param epochs: The number of training iterations.
        :type epochs: int, optional
        :param batch_size: The number of data samples to use in each iteration.
        :type batch_size: int, optional
        :param learning_rate: The initial learning rate for the optimizer.
        :type learning_rate: float, optional
        :param decay_rate: The decay rate used for the moving average of squared gradients.
        :type decay_rate: float, optional
        :param epsilon: A small constant to avoid division by zero.
        :type epsilon: float, optional
        :return: A callable that takes training inputs, training outputs, and model parameters as input and returns
            updated parameters and the training loss history.
        :rtype: callable
        """
        def callable(x_train, y_train, weights, bias):
            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=[2,3]))

            # History
            history = list()
            history.append(loss(x_train, y_train, weights, bias))

            # Initialize cumulated square gradient
            cumulated_square_grad = list()
            cumulated_square_grad.append(np.zeros_like(weights))
            cumulated_square_grad.append(np.zeros_like(bias))

            for epoch in range(epochs):
                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient
                grad_val = grad_loss(x_train[idxs,:], y_train[idxs,:], weights, bias)

                # Update cumulated square gradient of weights
                cumulated_square_grad[0] = decay_rate * cumulated_square_grad[0] + (1 - decay_rate) * grad_val[0] * grad_val[0]

                # Update weights
                weights = weights - learning_rate * grad_val[0] / (epsilon + np.sqrt(cumulated_square_grad[0]))

                # Update cumulated square gradient of bias
                cumulated_square_grad[1] = decay_rate * cumulated_square_grad[1] + (1 - decay_rate) * grad_val[1] * grad_val[1]

                # Update bias
                bias = bias - learning_rate * grad_val[1] / (epsilon + np.sqrt(cumulated_square_grad[1]))

                # Update history
                history.append(loss(x_train, y_train, weights, bias))
            return weights, bias, history
        return callable

    def fit(self, X=None, y=None, weights=None, bias=None, optimizer=None):
        """
        Fits the model to the data (X, y) using the specified parameters (weights and bias) and optimization
        strategy. The function applies the provided optimizer to train on the input data
        and returns the updated parameters and the loss function optimization history.

        The function does not modify the data or labels directly,
        but operates through the given optimizer.

        :param X: Input features for the model.
        :param y: Target labels corresponding to the input features.
        :param weights: Weights to optimize
        :param bias: Bias to optimize
        :param optimizer: Callable, the function used for optimization.
        :return: Optimized parameters and the loss function optimization history.
        """

        return optimizer(X, y, weights, bias)

    def predict(self, X=None, weights=None, bias=None):
        """
        Predicts output values based on the provided input data, model weights and bias.

        :param X: The input data, expected to be a NumPy array of shape (n_samples, n_features).
        :param weights: A list containing the weights of the model.
        :param bias: A float contain the bias of the model.
        :return: The predicted output values, as a NumPy array of shape (n_samples, n_outputs).
        """
        # Algorithm
        z = bias + X @ weights
        y_pred = jax.nn.sigmoid(-z)

        return y_pred