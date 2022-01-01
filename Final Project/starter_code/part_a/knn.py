from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    matrix = np.transpose(matrix)
    mat = nbrs.fit_transform(matrix)
    mat = np.transpose(mat)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # acc1 = knn_impute_by_user(sparse_matrix, val_data, 1)
    # acc6 = knn_impute_by_user(sparse_matrix, val_data, 6)
    # acc11 = knn_impute_by_user(sparse_matrix, val_data, 11)
    # acc16 = knn_impute_by_user(sparse_matrix, val_data, 16)
    # acc21 = knn_impute_by_user(sparse_matrix, val_data, 21)
    # acc26 = knn_impute_by_user(sparse_matrix, val_data, 26)
    #
    # k = [1, 6, 11, 16, 21, 26]
    # acc = [acc1, acc6, acc11, acc16, acc21, acc26]
    #
    # plt.plot(k, acc)
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    #
    # plt.show()
    #
    # high_k = k[acc.index(max(acc))]
    # print(high_k)
    # print(knn_impute_by_user(sparse_matrix, test_data, high_k))

    acc1 = knn_impute_by_item(sparse_matrix, val_data, 1)
    acc6 = knn_impute_by_item(sparse_matrix, val_data, 6)
    acc11 = knn_impute_by_item(sparse_matrix, val_data, 11)
    acc16 = knn_impute_by_item(sparse_matrix, val_data, 16)
    acc21 = knn_impute_by_item(sparse_matrix, val_data, 21)
    acc26 = knn_impute_by_item(sparse_matrix, val_data, 26)

    k = [1, 6, 11, 16, 21, 26]
    acc = [acc1, acc6, acc11, acc16, acc21, acc26]

    plt.plot(k, acc)
    plt.xlabel('k')
    plt.ylabel('accuracy')

    plt.show()

    high_k = k[acc.index(max(acc))]
    print(high_k)
    print(knn_impute_by_user(sparse_matrix, test_data, high_k))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
