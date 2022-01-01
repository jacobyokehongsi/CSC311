'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''
import matplotlib

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
        covariances[i] = np.matmul(np.transpose(x_minus_mu), x_minus_mu) / \
                         dig_lab_sample.shape[0] + 0.01 * np.identity(64)
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
        third_term_sub = np.matmul(np.linalg.inv(covariances[i]), np.transpose(x_minus_mu))
        third_term = 1 / 2 * np.diag(np.matmul(x_minus_mu, third_term_sub))
        likelihood[i] = -64 / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(np.linalg.det(covariances[i])) - third_term
    return np.transpose(likelihood)


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cond_like = np.zeros((digits.shape[0], 10))
    gen_like = generative_likelihood(digits, means, covariances)
    for i in range(digits.shape[0]):
        cond_like[i] = gen_like[i] - np.log(np.sum(np.exp(gen_like[i])))
    return cond_like


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    total_likelihood = 0
    for i in range(digits.shape[0]):
        total_likelihood += cond_likelihood[i][int(labels[i])]
    avg_likelihood = np.divide(total_likelihood, digits.shape[0])
    # Compute as described above and return
    return avg_likelihood


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(8, 8),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
        col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    print("Average Conditional Log Likelihood for Training Data:",
          avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("Average Conditional Log Likelihood for Testing Data:",
          avg_conditional_likelihood(test_data, test_labels, means, covariances))

    train_pred = classify_data(train_data, means, covariances)
    train_accuracy = np.sum(train_pred == train_labels) / len(train_labels)
    print("Training Accuracy:", train_accuracy)

    test_pred = classify_data(test_data, means, covariances)
    test_accuracy = np.sum(test_pred == test_labels) / len(test_labels)
    print("Testing Accuracy:", test_accuracy)

    for i in range(10):
        eigval, eigvec = np.linalg.eig(covariances[i])
        max_eigvac = eigvec[:, np.argmax(eigval)]
        plt.imshow(max_eigvac.reshape(8, 8), cmap='gray')
        # plt.show()
        plt.savefig("{}.png".format(i))

    # Evaluation


if __name__ == '__main__':
    main()
