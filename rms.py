import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return np.multiply(x, (1 - x))


def single_layer_forward_propagation(W, B, A_prev):
    Z = np.dot(W, A_prev) + B
    return logistic(Z)


def set_parameters(parameters, l, W, B):
    parameters["W" + str(l + 1)] = W
    parameters["b" + str(l + 1)] = B
    return parameters


def get_parameters(parameters, l):
    return parameters["W" + str(l + 1)], parameters["b" + str(l + 1)]


def vectorized_forward_propagation(X, parameters):
    W1, B1 = get_parameters(parameters, 0)
    A1 = single_layer_forward_propagation(W1, B1, np.transpose(X))

    W2, B2 = get_parameters(parameters, 1)
    A2 = single_layer_forward_propagation(W2, B2, A1)
    return A1, A2


def get_a(n_prev, n_next):
    return np.sqrt(2 / (n_prev + n_next))


def initialize_weights(X, hidden_neurons):
    parameters = {}
    a = get_a(X.shape[1], hidden_neurons)
    parameters["W1"] = np.random.uniform(
        size=(hidden_neurons, X.shape[1]), low=-a, high=a)
    parameters["b1"] = np.zeros((hidden_neurons, 1))
    a = get_a(hidden_neurons, 1)
    parameters["W2"] = np.random.uniform(
        size=(1, hidden_neurons), low=-a, high=a)
    parameters["b2"] = np.zeros((1, 1))
    return parameters


def vectorized_backward_propagation_with_rms(X, y, A1, A2, parameters, rmsprop, learning_rate, epsilon):
    """
    The above function implements the back propagation algorithm with the addition of RMSprop
    new_parameter = old_parameter - learning_rate / (sqrt(mean_squared_gradient) + epsilon) * gradient
    """

    m = X.shape[0]
    error = (A2 - y)

    # gamma = (1 - rate)
    gamma = 0.9  # as suggested by Hinton
    alpha = 0.1

    W2, B2 = get_parameters(parameters, 1)
    W2_fixed = W2
    delta2 = np.multiply(logistic_derivative(A2), error)

    B2_mean_squared = rmsprop["B2_mean_squared"]

    B2_gradient = 2 / m * np.dot(delta2, np.ones((m, 1)))

    B2_mean_squared = gamma * B2_mean_squared + \
        alpha * np.square(B2_gradient)

    B2 = B2 - learning_rate / \
        (np.sqrt(B2_mean_squared) + epsilon) * B2_gradient

    rmsprop["B2_mean_squared"] = B2_mean_squared

    W2_mean_squared = rmsprop["W2_mean_squared"]

    W2_gradient = 2 / m * np.dot(delta2, np.transpose(A1))

    W2_mean_squared = gamma * W2_mean_squared + \
        alpha * np.square(W2_gradient)

    W2 = W2 - learning_rate / \
        (np.sqrt(W2_mean_squared) + epsilon) * W2_gradient

    rmsprop["W2_mean_squared"] = W2_mean_squared

    parameters = set_parameters(parameters, 1, W2, B2)

    W1, B1 = get_parameters(parameters, 0)
    delta1 = np.multiply(logistic_derivative(
        A1), np.dot(np.transpose(W2_fixed), delta2))

    B1_mean_squared = rmsprop["B1_mean_squared"]
    B1_gradient = 2 / m * np.dot(delta1, np.ones((m, 1)))

    B1_mean_squared = gamma * B1_mean_squared + alpha * \
        np.square(B1_gradient)
    B1 = B1 - learning_rate / \
        (np.sqrt(B1_mean_squared) + epsilon) * B1_gradient

    rmsprop["B1_mean_squared"] = B1_mean_squared

    W1_mean_squared = rmsprop["W1_mean_squared"]
    W1_gradient = 2 / m * np.dot(delta1, X)

    W1_mean_squared = gamma * W1_mean_squared + \
        alpha * np.square(W1_gradient)
    W1 = W1 - learning_rate / \
        (np.sqrt(W1_mean_squared) + epsilon) * W1_gradient

    rmsprop["W1_mean_squared"] = W1_mean_squared

    parameters = set_parameters(parameters, 0, W1, B1)

    return parameters, rmsprop


def vectorized_backward_propagation(X, y, A1, A2, parameters, alpha):
    m = X.shape[0]
    error = (A2 - y)

    W2, B2 = get_parameters(parameters, 1)
    W2_fixed = W2
    delta2 = np.multiply(logistic_derivative(A2), error)
    B2 = B2 - alpha * 2 / m * np.dot(delta2, np.ones((m, 1)))
    W2 = W2 - alpha * 2 / m * np.dot(delta2, np.transpose(A1))
    parameters = set_parameters(parameters, 1, W2, B2)

    W1, B1 = get_parameters(parameters, 0)
    delta1 = np.multiply(logistic_derivative(
        A1), np.dot(np.transpose(W2_fixed), delta2))
    B1 = B1 - alpha * 2 / m * np.dot(delta1, np.ones((m, 1)))
    W1 = W1 - alpha * 2 / m * np.dot(delta1, X)
    parameters = set_parameters(parameters, 0, W1, B1)

    return parameters

def compute_loss(y, A2):
   
    return -np.mean(np.multiply(y, np.log(A2)) + np.multiply((1 - y), np.log(1 - A2)))

def get_mean_std(X):
    return np.mean(X, axis=0), np.std(X, axis=0)


def confusion_matrix(y_true, y_pred):
    classes = list(set(y_true))
    X = np.zeros((len(classes), len(classes)))
    for i in range(len(y_true)):
        X[classes.index(y_true[i]), classes.index(y_pred[i])] += 1
    return classes, X


def accuracy(X):
    return np.sum(np.diag(X)) / np.sum(X)


def precision(X):
    return np.diag(X) / np.sum(X, axis=0)


def recall(X):
    return np.diag(X) / np.sum(X, axis=1)


def normalization(X, means=None, sds=None):
    for j in range(X.shape[1]):
        if means is not None:
            X[:, j] = (X[:, j] - means[j])
        if sds is not None:
            X[:, j] = X[:, j] / sds[j]
    return X
