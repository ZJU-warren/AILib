import Tools.DistanceLib as DisLib
import numpy as np


def hierarchical(data, k, distance_type='min', method='euclidean', p=0):
    # init the cluster set
    cluster_num = data.shape[0]
    clusters = [np.array(data[i], ndmin=2) for i in range(cluster_num)]

    # loop until k clusters left
    for t in range(cluster_num, k, -1):
        # calculate the cluster-pair distance matrix
        distance_matrix = DisLib.calculate_distance_matrix_clusters(clusters, distance_type, method, p)

        # fill the diagonal element with the max value in matrix
        matrix_max_value = distance_matrix.max()
        for i in range(t):
            distance_matrix[i, i] = matrix_max_value
        # get the cluster-pair <i, j> with min distance
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # generate the new clusters
        temp_clusters = [clusters[x] for x in range(t) if x != i and x != j]
        temp_clusters.append(np.concatenate((clusters[i], clusters[j])))
        clusters = temp_clusters

    return clusters


if __name__ == '__main__':
    A = np.array(
        [[0, 21], [-7, 1], [2, 3], [9, 4], [3, 0]]
    )
    cluster = hierarchical(A, 3)
    for each in cluster:
        print(each)
