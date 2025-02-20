import jax
import jax.numpy as jnp

from utils.data_processer import *
from utils.models.base_model import BaseModel

class ANN(BaseModel):
    def __init__(self, layers_size=None, act_func=jnp.tanh, out_act_func=jax.nn.sigmoid, optimizer=None):
        """
        A class representing an artificial neural network configuration for specifying the structure,
        activation functions of each layer in the network and the optimization algorithm to use during training.

        :param layers_size: List specifying the size of each layer of the neural network.
            This includes both input and output layer sizes.
        :type layers_size: Optional[List[int]]

        :param act_func: Activation function to be applied to the intermediate layers
            of the neural network. Defaults to hyperbolic tangent (jnp.tanh).
        :type act_func: Callable

        :param out_act_func: Activation function to be applied to the output layer of the
            neural network. Defaults to sigmoid (jax.nn.sigmoid).
        :type out_act_func: Callable

        :param optimizer: Optimizer to be used for training the neural network.
        :type optimizer: dict
        """

        self.layers_size = layers_size
        self.act_func = act_func
        self.out_act_func = out_act_func
        self.params = None

        # Check if regularization is required
        if 'penalization' in optimizer.keys():
            optimizer['loss_function'] = self.regularized_loss(penalization=optimizer['penalization'])
        else:
            optimizer['loss_function'] = self.cross_entropy

        # Set optimization algorithm
        ann_params = {k: v for k, v in optimizer.items() if k != 'opt_type' and k != 'penalization'}
        if optimizer['opt_type'] == 'SGD':
            self.optimizer = self.SGD(**ann_params)
        elif optimizer['opt_type'] == 'SGD_momentum':
            self.optimizer = self.SGD_momentum(**ann_params)
        elif optimizer['opt_type'] == 'NAG':
            self.optimizer = self.NAG(**ann_params)
        elif optimizer['opt_type'] == 'RMSprop':
            self.optimizer = self.RMSprop(**ann_params)

    def initialize_parameters(self, layers_size):
        """
            Returns the parameters of the artificial neural network given the number of
            neurons in its layers. Specifically, it sets the matrix of weights and the
            bias vector for each layer, initialized randomly.

            :param layers_size: Ordered sizes of the layers of the artificial neural network.
            :type layers_size: list
            :return: Parameters of the artificial neural network, specifically weights and biases.
            :rtype: list
        """

        layers_size = jnp.array(layers_size)
        np.random.seed(0)  # For reproducibility
        self.layers_size = layers_size

        params = list()
        for i in range(len(self.layers_size) - 1):
            W = np.random.randn(self.layers_size[i + 1], self.layers_size[i])
            b = np.zeros((self.layers_size[i + 1], 1))

            params.append(W)
            params.append(b)

        return params

    def MSW(self, params):
        """
        Computes the mean squared weights (MSW) of an artificial neural network's weights.

        This function calculates the average of the squared values of the weights present
        in the given parameters of an artificial neural network. Typically, the weights
        and biases are passed as a list, where this function processes only the weights.

        :param params: Parameters of the artificial neural network, usually alternating
            weights and biases
        :type params: list
        :return: Mean of squared weights of the artificial neural network
        :rtype: float
        """

        # Extract weights
        weights = params[::2]

        # Calculate MSW
        partial_sum = 0.0
        n_weights = 0
        for W in weights:
            partial_sum = partial_sum + jnp.sum(W * W)
            n_weights = n_weights + W.size

        return partial_sum / n_weights

    # Loss functions
    def cross_entropy(self, x, y, params):
        """
        Defines the cross-entropy loss function for classification problems with
        logistic output. This function calculates the negative log-likelihood
        of the predictions made by the model when compared with the true labels.
        It is commonly used as a loss function for binary classification tasks.

        :returns: the value cross-entropy loss
            for given input features, true labels, and model parameters.
        :rtype: float
        """

        y_pred = self.get_prediction(x, params)
        return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

    def regularized_loss(self, penalization):
        """
        Computes a regularized loss function by combining the cross entropy function with
        a penalization term. The penalization term is scaled based on the dataset size
        and involves the Mean Squared Weight (MSW) for the provided parameters.

        This method returns a callable function that takes `x`, `y`, and `params` as inputs
        and computes the regularized loss by applying the cross entropy function and adding
        the scaled penalization term.

        :param penalization: A scalar value representing the penalization term to be
            applied during regularization. This value is combined with the MSW.
        :return: Returns a callable function which computes the regularized loss as a
            combination of the cross entropy function and the penalization term.
        """
        def callable(x, y, params):
            return self.cross_entropy(x, y, params) + penalization / (2 * x.shape[0]) * self.MSW(params)

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

        def callable(x_train, y_train, params):
            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=2))

            # History
            history = list()
            history.append(loss(x_train, y_train, params))

            for epoch in range(epochs):
                # Get learning rate
                learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))

                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient
                grad_val = grad_loss(x_train[idxs, :], y_train[idxs, :], params)

                # Update params
                for i in range(len(params)):
                    params[i] = params[i] - learning_rate * grad_val[i]

                # Update history
                history.append(loss(x_train, y_train, params))
            return params, history

        return callable

    def SGD_momentum(
            self,
            loss_function,
            epochs=1000,
            batch_size=128,
            learning_rate_min=1e-3,
            learning_rate_max=1e-1,
            learning_rate_decay=1000,
            momentum=0.9,
    ):
        """
        Creates a Stochastic Gradient Descent (SGD) with Momentum optimization as a callable function.

        The function minimizes the specified loss function by iteratively updating the model parameters
        using mini-batches of training data.
        Velocity vectors are employed to incorporate momentum in the optimization process,
        which helps in faster convergence and avoidance of local minima.

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
        :param momentum: The momentum coefficient used to regulate the influence
            of the previous gradient updates on the current update. Default is 0.9.

        :return: Returns a callable function that takes training data and initial
            model parameters as input and outputs optimized parameters and the
            optimization history.
        """

        def callable(x_train, y_train, params):
            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=2))

            # History
            history = list()
            history.append(loss(x_train, y_train, params))

            # Initialize velocity
            velocity = list()
            for i in range(len(params)):
                velocity.append(np.zeros_like(params[i]))

            for epoch in range(epochs):
                # Get learning rate
                learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))

                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient
                grad_val = grad_loss(x_train[idxs, :], y_train[idxs, :], params)

                for i in range(len(params)):
                    # Compute velocity[i]
                    velocity[i] = momentum * velocity[i] - learning_rate * grad_val[i]

                    # Update params[i]
                    params[i] = params[i] + velocity[i]

                # Update history
                history.append(loss(x_train, y_train, params))
            return params, history

        return callable

    def NAG(
            self,
            loss_function,
            epochs=1000,
            batch_size=128,
            learning_rate_min=1e-3,
            learning_rate_max=1e-1,
            learning_rate_decay=1000,
            momentum=0.9,
    ):
        """
        Implements the Nesterov Accelerated Gradient (NAG) optimization training method.
        This method is used to iteratively optimize an artificial neural network by minimizing the loss between the
        predictions and the correct target values.

        The class provides a callable which trains the model parameters using the provided loss function
        and hyperparameters. NAG includes an additional momentum term to accelerate convergence and
        reduce oscillations in gradient descent.

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
        :param momentum: The momentum coefficient used to regulate the influence
            of the previous gradient updates on the current update. Default is 0.9.

        :return: Returns a callable function that takes training data and initial
            model parameters as input and outputs optimized parameters and the
            optimization history.
        """
        def callable(x_train, y_train, params):

            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=2))

            # History
            history = list()
            history.append(loss(x_train, y_train, params))

            # Initialize velocity
            velocity = list()
            for i in range(len(params)):
                velocity.append(np.zeros_like(params[i]))

            for epoch in range(epochs):
                # Get learning rate
                learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))

                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient:
                # here it's necessary to calculate the arguments that will substitute 'params' on gradient evaluation
                grad_args = list()
                for i in range(len(params)):
                    grad_args.append(np.zeros_like(params[i]))
                for i in range(len(params)):
                    grad_args[i] = params[i] - momentum * velocity[i]

                grad_val = grad_loss(x_train[idxs, :], y_train[idxs, :], grad_args)

                for i in range(len(params)):
                    # Compute velocity[i]
                    velocity[i] = momentum * velocity[i] + learning_rate * grad_val[i]

                    # Update params[i]
                    params[i] = params[i] - velocity[i]

                # Update history
                history.append(loss(x_train, y_train, params))
            return params, history

        return callable

    def RMSprop(
            self,
            loss_function,
            epochs=1000,
            batch_size=128,
            learning_rate=0.1,
            decay_rate=0.9,
            epsilon=1e-8
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

        def callable(x_train, y_train, params):
            # Number of samples
            num_samples = x_train.shape[0]

            # Loss and it's gradient functions
            loss = jax.jit(loss_function)
            grad_loss = jax.jit(jax.grad(loss_function, argnums=2))

            # History
            history = list()
            history.append(loss(x_train, y_train, params))

            # Initialize cumulated square gradient
            cumulated_square_grad = list()
            for i in range(len(params)):
                cumulated_square_grad.append(np.zeros_like(params[i]))

            for epoch in range(epochs):
                # Select batch_size indices randomly
                idxs = np.random.choice(num_samples, batch_size)

                # Calculate gradient
                grad_val = grad_loss(x_train[idxs, :], y_train[idxs, :], params)

                for i in range(len(params)):
                    # Update cumulated square gradient
                    cumulated_square_grad[i] = decay_rate * cumulated_square_grad[i] + (1 - decay_rate) * grad_val[i] * \
                                               grad_val[i]

                    # Update params[i]
                    params[i] = params[i] - learning_rate * grad_val[i] / (epsilon + np.sqrt(cumulated_square_grad[i]))

                # Update history
                history.append(loss(x_train, y_train, params))
            return params, history

        return callable
    
    def get_prediction(self, X=None, params=None):
        """
        Predicts output values based on the provided input data and model parameters. Applies the defined activation
        functions to intermediary layers and output layer. The function assumes the
        weights and biases are provided sequentially in the params list.

        :param X: The input data, expected to be a NumPy array of shape (n_samples, n_features).
        :param params: A list containing the weights and biases for each layer of the ANN, alternating
                       between weight matrices and bias vectors.
        :return: The predicted output values, as a NumPy array of shape (n_samples, n_outputs).
        """

        # Number of ANN layers
        num_layers = int(len(self.layers_size)) + 1

        # Algorithm
        layer = X.T
        weights = params[0::2]
        biases = params[1::2]
        for i in range(num_layers - 2):
            # Update layer values
            layer = weights[i] @ layer + biases[i]

            # Apply activation function
            layer = self.act_func(layer)

        # On the output layer it is applied the sigmoid function
        # since the output is needed to be between 0 and 1
        layer = self.out_act_func(layer)
        layer = layer.T

        return layer

    def fit(self, X=None, y=None):
        """
        Fits the model to the data (X, y) using the optimization strategy.
        The function applies the provided optimizer to train on the input data
        and returns the updated parameters and the loss function optimization history.

        The function does not modify the data or labels directly,
        but operates through the given optimizer.

        :param X: Input features for the model.
        :param y: Target labels corresponding to the input features.
        :return: the loss function optimization history.
        """

        # Set first and last layer size (this is useful for GA)
        self.layers_size[0] = X.shape[1]
        self.layers_size[-1] = 1

        # Initialize parameters: weights and biases
        self.params = self.initialize_parameters(self.layers_size)

        self.params, history = self.optimizer(X, y, self.params)
        return history

    def predict(self, X=None):
        """
        This is an entry point for the ANN model prediction routine.

        :param X: Input features for the model.
        :return: Predicted output values for the input features.
        """
        return self.get_prediction(X, self.params), None