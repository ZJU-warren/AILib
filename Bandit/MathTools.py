import numpy as np


def vec(matrix):
    return matrix.reshape(-1, 1)


def mat(vector, N):
    return vector.reshape(-1, N)


def join(a, b):
    return np.concatenate([a, b])
