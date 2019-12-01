import numpy as np


# calculate point-pair distance
def calculate_distance_point(a, b, method='euclidean', p=0):

    # Manhattan Distance
    def distance_manhattan(a, b):
        return np.sum(np.abs(a - b))

    # Euclidean Distance
    def distance_euclidean(a, b):
        return np.sqrt((a - b).T.dot((a - b)))

    # Chebyshev Distance
    def distance_chebyshev(a, b):
        return np.max(np.abs(a - b))

    # Minkowski Distance
    def distance_minkowski(a, b, p):
        return (np.sum(np.abs(a - b) ** p)) ** (1 / p)

    # Cosine Distance
    def distance_cosine(a, b):
        return a.dot(b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if p != 0:
        return distance_minkowski(a, b, p)
    if method == 'manhattan':
        return distance_manhattan(a, b)
    if method == 'euclidean':
        return distance_euclidean(a, b)
    if method == 'chebyshev':
        return distance_chebyshev(a, b)
    if method == 'cosine':
        return distance_cosine(a, b)


# calculate cluster-pair distance
def calculate_distance_cluster(Gp, Gq, distance_type, method='euclidean', p=0):
    def get_linkage(Gp, Gq, method, p, func, base_value=-1):
        if base_value == -1:
            distance = calculate_distance_point(Gp[0], Gq[0], method, p)
        else:
            distance = base_value

        for eachI in Gp:
            for eachJ in Gq:
                distance = func(distance, calculate_distance_point(eachI, eachJ, method, p))
        return distance

    def min_distance(Gp, Gq, method, p):
        return get_linkage(Gp, Gq, method, p, min)

    def max_distance(Gp, Gq, method, p):
        return get_linkage(Gp, Gq, method, p, max)

    def center_distance(Gp, Gq, method, p):
        return calculate_distance_point(np.mean(Gp, axis=0), np.mean(Gq, axis=0), method, p)

    def mean_distance(Gp, Gq, method, p):
        return get_linkage(Gp, Gq, method, p, lambda x, y: x + y, 0) / (Gp.shape[0] * Gq.shape[0])

    if distance_type == 'min':
        return min_distance(Gp, Gq, method, p)
    if distance_type == 'max':
        return max_distance(Gp, Gq, method, p)
    if distance_type == 'center':
        return center_distance(Gp, Gq, method, p)
    if distance_type == 'mean':
        return mean_distance(Gp, Gq, method, p)


# calculate cluster-pair distance matrix
def calculate_distance_matrix_clusters(clusters, distance_type='min', method='euclidean', p=0):
    """
    :param clusters: a list contains many clusters whose elements are vectors
    :return: distance matrix [dij] where dij means the distance between cluster i and j
    """
    cluster_num = len(clusters)
    distance_matrix = np.zeros((cluster_num, cluster_num))
    for i in range(cluster_num):
        for j in range(i+1, cluster_num):
            distance_matrix[i, j] = distance_matrix[j, i] = \
                calculate_distance_cluster(clusters[i], clusters[j], distance_type, method, p)
    return distance_matrix
