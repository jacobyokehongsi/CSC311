'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # print((data.get_digits_by_label(train_data, train_labels, 1).shape))
    for i in range(10):
        dig_lab = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(dig_lab, axis=0)
    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        dig_lab_sample = data.get_digits_by_label(train_data, train_labels, i)
        x_minus_mu = dig_lab_sample - means[i]
        # print(x_minus_mu.shape[0])
        covariances[i] = np.matmul(np.transpose(x_minus_mu), x_minus_mu)
        # covariances[i] = np.matmul(np.transpose(x_minus_mu), x_minus_mu) / len(x_minus_mu)
    # print(covariances)
    # Compute covariances
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    likelihood = np.zeros((10, digits.shape[0]))
    for i in range(10):
        x_minus_mu = digits - means[i]
        # print(np.linalg.inv(covariances[i]).shape)
        # print(x_minus_mu.shape)
        third_term_sub = np.matmul(np.linalg.inv(covariances[i]), np.transpose(x_minus_mu))
        # print(third_term_sub.shape)
        third_term = 1/2 * np.sum(np.matmul(x_minus_mu, third_term_sub), axis=0)
        # thirdss = -1/2 * np.sum(np.dot((digits-means[i]), np.linalg.solve(covariances[i], (digits-means[i]).T)), axis=0)
        # print("trent", thirdss.shape)
        # print(third_term.shape)
        likelihood[i] = -64/2 * np.log(2 * np.pi) - 1/2 * np.log(np.linalg.det(covariances[i])) - third_term
    return np.transpose(likelihood)

def generative_likelihood2(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    likelihood = np.zeros((10, digits.shape[0]))
    for i in range(10):
        first = -64/2 * np.log(2 * np.pi)
        second = - 1/2 * np.log(np.linalg.det(covariances[i]))
        # x_minus_mu = digits - means[i]
        # third_term_sub = np.matmul(np.linalg.inv(covariances[i]), np.transpose(x_minus_mu))
        # third_term = 1/2 * np.sum(np.matmul(x_minus_mu, third_term_sub), axis=0)
        thirdss = -1/2 * np.sum(np.dot((digits-means[i]), np.linalg.solve(covariances[i], (digits-means[i]).T)), axis=0)
        likelihood[i] = first + second + thirdss
    return np.transpose(likelihood)


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    cov = compute_sigma_mles(train_data, train_labels)
    mean = compute_mean_mles(train_data, train_labels)
    print(generative_likelihood(train_data, mean, cov))
    print("==========================")
    print(generative_likelihood2(train_data, mean, cov))
    # main()
