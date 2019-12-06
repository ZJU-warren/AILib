import numpy as np
import Tools.DistanceLib as DisLib
from Tools.Heap import Heap

class Node:
    # node's pointers
    lson = None
    rson = None
    father = None

    # node's info
    def __init__(self, data_index):
        self.data_index = data_index

    def get_another(self, son):
        return self.rson if son is self.lson else self.lson


class KNN:
    root = None

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N, self.k = X.shape
        self.root = self.construct_tree(0, [i for i in range(self.N)])

    # construct the KD-tree by divide and conquer
    def construct_tree(self, axis, id_set):
        if len(id_set) == 0:
            return None

        # generated the arg-sort index list of X[id_set][axis]
        value_list = self.X[id_set, axis]
        sort_idx_list = np.argsort(value_list)

        # split the data
        middle_idx = len(id_set) // 2
        left_list = [id_set[i] for i in sort_idx_list[:middle_idx]]
        right_list = [id_set[i] for i in sort_idx_list[middle_idx+1:]]

        # create a new node
        node = Node(id_set[sort_idx_list[middle_idx]])
        axis_new = (axis + 1) % self.k

        # divide remain data
        node.lson = self.construct_tree(axis_new, left_list)
        node.rson = self.construct_tree(axis_new, right_list)

        return node

    # find the x's nearest neighbor (in the given tree(root))
    def locate_nearest(self, root, x, axis):
        # if tree is empty
        if root is None:
            return None, -1

        # calculate the axis in next layer
        axis_new = (axis + 1) % self.k

        # 1. decide the sub-tree to search
        if x[axis] <= self.X[root.data_index, axis]:
            son = root.lson
        else:
            son = root.rson

        # 2. calculate the nearest in the sub-tree
        target, distance = self.locate_nearest(son, x, axis_new)

        # 3. if target is None or exist area joint
        if target is None or \
                abs(x[axis] - self.X[target.data_index, axis]) < abs(self.X[target.data_index, axis]
                                                                     - self.X[root.data_index, axis]):
            sub_target, sub_distance = self.locate_nearest(root.get_another(son), x, axis_new)
            if sub_target is not None and sub_distance < distance:
                target, distance = sub_target, sub_distance

        # 4. compare with the root
        root_x_distance = DisLib.calculate_distance_point(self.X[root.data_index], x)
        if target is None or root_x_distance < distance:
            target, distance = root, root_x_distance
        # print('axis = ', axis, ':', self.X[target.data_index])
        return target, distance
    #
    # # find the x's k-nearest neighbor (in the given tree(root))
    # def locate_k_nearest(self, root, x, k=1, axis=0):
    #     # default the distances between heap_i and x is ascending
    #     heap = Heap(lambda a, b: a > b)
    #
    #     # if tree is empty
    #     if root is None:
    #         return heap
    #
    #     # calculate the axis in next layer
    #     axis_new = (axis + 1) % self.k
    #
    #     # 1. decide the sub-tree to search
    #     if x[axis] <= self.X[root.data_index, axis]:
    #         son = root.lson
    #     else:
    #         son = root.rson
    #
    #     # 2. calculate the nearest in the sub-tree
    #     target = self.locate_k_nearest(son, x, k, axis_new)
    #
    #     # 3. if target is None or exist area joint
    #     if target.cnt == 0 or\
    #             abs(self.X[target.top().info.data_index, axis] - self.X[root.data_index, axis]) < target.top().key:
    #         sub_target = self.locate_k_nearest(root.get_another(son), x, k, axis_new)
    #         # if sub_target is not None and sub_distance < target.top():
    #         #     target, distance = sub_target, sub_distance
    #         target = target.merge(sub_target, k)
    #     # 4. compare with the root
    #
    #     root_x_distance = DisLib.calculate_distance_point(self.X[root.data_index], x)
    #     if target.cnt == 0 or root_x_distance < target.top().key:
    #         target = root, root_x_distance
    #     print('axis = ', axis, ':', self.X[target.data_index])
    #     return target

    # predict the label of x
    def predict(self, x):
        nearest, distance = self.locate_nearest(self.root, x, 0)
        print(self.X[nearest.data_index])

    # output the tree structure
    def print_tree(self, root, deep=0):
        if root is None:
            return
        self.print_tree(root.lson, deep+1)
        print('---' * deep, self.X[root.data_index, :])
        self.print_tree(root.rson, deep+1)


if __name__ == '__main__':
    X = np.array(
        [[2, 3], [9, 6], [5, 4],
         [4, 7], [8, 1], [7, 2]])

    y = np.array([1, 1, -1, 1, 1, 1])
    knn_clf = KNN(X, y)
    # knn_clf.print_tree(knn_clf.root)
    knn_clf.predict(np.array([2, 4.5]))
