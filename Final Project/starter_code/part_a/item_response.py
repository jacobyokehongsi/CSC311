from matplotlib import pyplot as plt

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # theta = user's ability
    # beta = question difficulty

    user_list = np.array(data.get("user_id"))
    question_list = np.array(data.get("question_id"))
    correct_list = np.array(data.get("is_correct"))
    # print('SUP USP', user_list)

    log_lklihood = 0
    for i in range(len(user_list)):
        # log_lklihood = np.matmul(correct_list[i], np.log(sigmoid((theta[user_list]-beta[question_list]).sum()))) + \
        #                np.matmul(1 - correct_list, np.log(1 - sigmoid(theta[user_list]-beta[question_list])))

        log_lklihood += correct_list[i] * (theta[user_list[i]] - beta[question_list[i]]) - \
                        np.log(1 + np.exp(theta[user_list[i]]-beta[question_list[i]]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_list = np.array(data.get("user_id"))
    question_list = np.array(data.get("question_id"))
    correct_list = np.array(data.get("is_correct"))

    for i in range(len(user_list)):
        sig = sigmoid((theta[user_list[i]] - beta[question_list[i]]).sum())
        theta[user_list[i]] = theta[user_list[i]] + lr * (correct_list[i] - sig)
        beta[question_list[i]] = beta[question_list[i]] + lr * (sig - correct_list[i])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    user_list = data.get("user_id")
    question_list = data.get("question_id")
    theta = np.zeros(len(user_list)+1)
    beta = np.zeros(len(question_list)+1)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_lld = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        # test_lld = neg_log_likelihood(data=test_data, theta=theta, beta=beta)

        train_lld_lst.append(np.sum(-neg_lld))
        val_lld_lst.append(np.sum(-val_lld))

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # print(train_data)

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # theta1, beta1, val_acc_lst1, train_lld_lst1, val_lld_lst1 = irt(train_data, val_data, 0.1, 60)
    # theta2, beta2, val_acc_lst2, train_lld_lst2, val_lld_lst2 = irt(train_data, val_data, 0.01, 30)
    theta3, beta3, val_acc_lst3, train_lld_lst3, val_lld_lst3 = irt(train_data, val_data, 0.1, 60)
    # theta4, beta4, val_acc_lst4, train_lld_lst4, val_lld_lst4 = irt(train_data, val_data, 0.1, 30)

    # print("validation rate accuracy 1:", val_acc_lst1)
    # print("validation rate accuracy 2:", val_acc_lst2)
    print("validation rate accuracy 3:", val_acc_lst3)
    # print("validation rate accuracy 4:", val_acc_lst4)

    plt.figure()
    plt.plot(train_lld_lst3, label="training log-likelihoods")
    plt.plot(val_lld_lst3, label="validation log-likelihoods")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihoods")
    plt.title("training and validation log-likelihoods as a function of iterations")
    plt.legend()
    plt.savefig("train and val log-likelihood.png")

    plt.figure()
    plt.plot(val_acc_lst3, label="validation accuracy")

    # part (c)
    val_acc = evaluate(val_data, theta3, beta3)
    test_acc = evaluate(test_data, theta3, beta3)

    print("Validation Accuracy:", val_acc)
    print("Test Accuracy:", test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #

    j1 = sigmoid(np.sort(theta3) - beta3[0])
    j2 = sigmoid(np.sort(theta3) - beta3[1])
    j3 = sigmoid(np.sort(theta3) - beta3[2])

    plt.figure()
    plt.plot(np.sort(theta3), j1, label="Question 1")
    plt.plot(np.sort(theta3), j2, label="Question 2")
    plt.plot(np.sort(theta3), j3, label="Question 3")
    plt.title("Probability of Correct Response as a function of Theta given a Question j")
    plt.ylabel("Probability of Correct Response")
    plt.xlabel("Theta given Question j")
    plt.legend()
    plt.savefig("Probability of Correct Response.png")

    plt.show()

    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
