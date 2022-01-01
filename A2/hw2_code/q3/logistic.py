from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept). [w1, w2 ,w3, b1]
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    bias_dummy = np.ones((len(data), 1))  # column of 1s for the bias terms in matrix multiplication
    x = np.concatenate((data, bias_dummy), axis=1)
    z = np.matmul(x, weights)  # z = xw
    y = sigmoid(z)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################

    tT = np.transpose(targets)
    oneminus_tT = np.transpose(1 - targets)
    log_y = np.log2(y)
    oneminus_log_y = np.log2(1 - y)

    N = len(y)
    ce = np.sum((-np.matmul(tT, log_y) - np.matmul(oneminus_tT, oneminus_log_y))) / N

    correct = 0
    for i in range(N):
        if targets[i][0] == 1:
            if y[i][0] >= 0.5:
                correct += 1
        if targets[i][0] == 0:
            if y[i][0] < 0.5:
                correct += 1
    frac_correct = correct / float(N)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f, frac_correct = evaluate(targets, y)

    bias_dummy = np.ones((len(data), 1))  # column of 1s for the bias terms in matrix multiplication
    x = np.concatenate((data, bias_dummy), axis=1)
    y_minus_t = y - targets
    df = np.transpose(np.matmul(np.transpose(y_minus_t), x)) / data.shape[0]  # Divide by dim to get df for 1 data pt.
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
