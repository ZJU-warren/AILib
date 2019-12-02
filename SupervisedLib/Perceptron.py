"""
    preceptron algorithm will be converge iff training data can be linearly separated
"""

import numpy as np
from Tools import mathLib


# perceptron algorithm (dual version, kernel trick will be supported)
def perceptron(X, y, learning_ratio, kernel=mathLib.calculate_gram_matrix):
    num_row, num_col = X.shape

    # step 1: calculate the gram/kernel matrix of X
    gram_matrix = kernel(X)

    # step 2: init the parameters
    alpha = np.zeros(num_row)
    b = 0

    # step 3: fit the training data (X, y)
    flag_wrong_classify = True

    # loop until converge, which means all the samples can be classify correctly
    while flag_wrong_classify:
        flag_wrong_classify = False

        # traversal all samples once
        for i in range(num_row):
            # loop until fix the sample
            while y[i] * ((alpha * y).dot(gram_matrix[:, i]) + b) <= 0:
                # update the parameters
                alpha[i] += learning_ratio
                b += learning_ratio * y[i]

                # set the flag
                flag_wrong_classify = True

    # calculate the super-plane
    w = (alpha * y).dot(X)
    b = alpha.dot(y)
    return w, b


# predict sample's label
def predict(w, b, x):
    # point on the super-plane will be classified as positive sample
    return 1 if np.sign(w.dot(x) + b) >= 0 else -1


if __name__ == '__main__':
    X = np.array([
        [3, 3],
        [4, 3],
        [1, 1]
    ])
    y = np.array([1, 1, -1])
    w, b = perceptron(X, y, 1)
    print(w, b)

    x = np.array([0, 3])
    print(predict(w, b, x))
