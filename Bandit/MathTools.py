import numpy as np


def vec(matrix):
    return matrix.reshape(-1, 1)


def mat(vector, N):
    return vector.reshape(-1, N)


def join(a, b):
    return np.concatenate([a, b])


if __name__ == '__main__':
    # X = np.array([[1, 3, 4], [4, 41, 1]]).T
    # a = np.array([[5, 6], [1, 3]]).T
    a1 = np.array([1, 3, 1])
    b = np.array([4, 5, 1])
    # print(join(a1, b))
    # h = [1, 14, 41, 52, 5]
    # print(np.argmax(h))
    X = np.zeros((6, 7))
    X[:, 2] = join(a1, b)
    print(X)
