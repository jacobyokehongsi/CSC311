from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    # LR = 0.1, 0.03
    # iters = 500, 200

    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 500
    }
    weights = np.zeros((M+1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train_loss = []
    valid_loss = []

    for t in range(hyperparameters["num_iterations"]):
        # training the data
        train_f, train_df, train_y = logistic(weights, train_inputs, train_targets, hyperparameters)
        train_ce, train_frac_correct = evaluate(train_targets, train_y)
        train_loss.append(train_ce)
        weights = weights - (hyperparameters["learning_rate"] * train_df)

        # validating the data
        valid_y = logistic_predict(weights, valid_inputs)
        valid_ce, valid_frac_correct = evaluate(valid_targets, valid_y)
        valid_loss.append(valid_ce)

    print("Dataset: mnist_train")
    print("Learning Rate:", hyperparameters["learning_rate"])
    print("Num Iterations:", hyperparameters["num_iterations"])
    print("Train Cross Entropy:", train_ce)
    print("Train Correct %:", train_frac_correct * 100)
    print("Valid Cross Entropy:", valid_ce)
    print("Valid Correct %:", valid_frac_correct * 100)

    plt.figure(0)
    plt.title("Training and Valid Cross Entropy vs. Iteration (mnist_train)")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(train_loss, label="Training")
    plt.plot(valid_loss, label="Validation")

    plt.legend()
    plt.show()

    test_loss = []
    test_inputs, test_targets = load_test()
    test_y = logistic_predict(weights, test_inputs)
    test_ce, test_frac_correct = evaluate(test_targets, test_y)
    test_loss.append(test_ce)

    print("Test Cross Entropy:", test_ce)
    print("Test Correct %:", test_frac_correct * 100)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    np.random.seed(0)
    run_logistic_regression()
    # run_pen_logistic_regression()
