from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    # (number of correctly predicted cases, divided by total number of data points)
    k = [1, 3, 5, 7, 9]
    valid_class_rate = []

    for i in k:
        valid_pred = knn(i, train_inputs, train_targets, valid_inputs)
        correct = 0
        for j in range(len(valid_pred)):
            if valid_pred[j][0] == valid_targets[j][0]:
                correct += 1

        total = len(valid_pred)

        classification_rate = correct / total
        valid_class_rate.append(classification_rate)

    # print(valid_class_rate)

    plt.figure(0)
    plt.scatter(k, valid_class_rate)
    plt.title("Classification Rate on the Validation Set vs. k")
    plt.xlabel("k")
    plt.ylabel("Classification Rate")
    plt.savefig("q3.1a.jpg")

    # print(np.argmax(valid_class_rate))
    # print(valid_class_rate[1] == valid_class_rate[3])
    # print(valid_class_rate[1] == valid_class_rate[2])

    test_class_rate = []
    for i in k:
        test_pred = knn(i, train_inputs, train_targets, test_inputs)
        correct = 0
        for j in range(len(test_pred)):
            if test_pred[j][0] == test_targets[j][0]:
                correct += 1

        total = len(test_pred)

        classification_rate = correct / total
        test_class_rate.append(classification_rate)

    # print(test_class_rate)

    plt.figure(1)
    plt.scatter(k, test_class_rate)
    plt.title("Classification Rate on the Test Set vs. k")
    plt.xlabel("k")
    plt.ylabel("Classification Rate")
    plt.savefig("q3.1b.jpg")

    # print(np.argmax(test_class_rate))
    # print(test_class_rate[2] == test_class_rate[3])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
