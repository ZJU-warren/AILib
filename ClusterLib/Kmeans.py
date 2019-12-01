import Tools.DistanceLib as DisLib
import numpy as np


# calculate the center point when clusters given
def calculate_center(data, point_label, k):
    num_row, num_col = data.shape

    # calculate the new center_points
    temp_center_points = np.zeros((k, num_col))
    temp_counts = np.array([[0]] * k)
    for i in range(num_row):
        temp_counts[point_label[i]] += 1
        temp_center_points[point_label[i]] += data[i]

    # fill the empty clusters counts
    for i in range(k):
        if temp_counts[i] == 0:
            temp_counts[i] = 1

    # calculate the new centers
    center_points = temp_center_points / temp_counts
    return center_points


# KMeans algorithm
def kmeans(data, k, rounds=100, show_log=False):
    # init and min-max normalization
    num_row, num_col = data.shape
    attribute_max = np.max(data, axis=0)
    attribute_min = np.min(data, axis=0)
    data = (data - attribute_min) / (attribute_max - attribute_min)

    # init the center points by random
    center_points = np.random.random((k, num_col))
    # init the point_label and the label_cnt
    point_label = [-1] * num_row
    label_cnt = [0] * k

    # step1: loop t rounds
    for t in range(rounds):
        # 1. classify the points to k cluster centered by center_points
        for i in range(num_row):
            distances = [DisLib.calculate_distance_point(data[i], center_points[j]) for j in range(k)]
            point_label[i] = int(np.argmin(distances))
            label_cnt[point_label[i]] += 1

        # 2. generate the distance between point and center they clustered
        temp_center_points = calculate_center(data, point_label, k)
        distance_to_center = np.array(
            [DisLib.calculate_distance_point(data[i], temp_center_points[point_label[i]]) for i in range(num_row)])
        idx_after_sort = np.argsort(-distance_to_center)

        # 3. replace the empty cluster by the points which are most deviate from the center they belong to
        replace_idx = 0
        for i in range(k):
            if label_cnt[i] == 0:
                while label_cnt[point_label[idx_after_sort[replace_idx]]] <= 1:
                    replace_idx += 1
                label_cnt[point_label[idx_after_sort[replace_idx]]] -= 1
                point_label[idx_after_sort[replace_idx]] = i
                replace_idx += 1
        temp_center_points = calculate_center(data, point_label, k)

        # 4. if centers have no change, algorithm ends up
        if (temp_center_points - center_points == np.zeros((k, num_col))).all():
            break
        else:
            center_points = temp_center_points

        # 5. calculate the loss
        distance_to_center = np.array(
            [DisLib.calculate_distance_point(data[i], center_points[point_label[i]]) for i in range(num_row)])
        loss = distance_to_center.sum()
        if show_log:
            print('after iteration %d, total loss is %f' % (t, loss))

    # step2: generate the clusters
    clusters = [None for _ in range(k)]
    for i in range(num_row):
        recover_data = data[i] * (attribute_max - attribute_min) + attribute_min
        recover_data = np.array(recover_data, ndmin=2)
        if clusters[point_label[i]] is None:
            clusters[point_label[i]] = recover_data
        else:
            clusters[point_label[i]] = np.concatenate((clusters[point_label[i]], recover_data))
    return clusters


if __name__ == '__main__':
    A = np.array(
        [[0, 21], [-7, 1], [2, 3], [9, 4], [3, 0]]
    )
    cluster = kmeans(A, 3, 100, show_log=True)
    for each in cluster:
        print(each)
