from MathTools import *
from FactorItem import FactorItem


class FactorUCB:
    def __init__(self, d, l, N, W,
                 lambda_1, lambda_2, alpha_u, alpha_a):             # line 1
        # basic setting
        self.d = d; self.l = l;  self.N = N
        self.W = W
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_u = alpha_u
        self.alpha_a = alpha_a

        # init the parameters                                           # line 2
        self.A = lambda_1 * np.identity(d+l)
        self.b = np.zeros((d+l)*N)
        self.theta = np.linalg.inv(self.A).dot(self.b)

        # arms pool
        self.items_pool = {}

        # opt the calculation
        self.AI = np.linalg.inv(self.A)
        self.WT = W.T
        self.mat_theta_Wu = None

        # recorder
        self.rec_item = None

    def decide(self, item_list, item_xs, user):                         # line 3, 4
        self.mat_theta_Wu = mat(self.theta, self.N).dot(self.W[user, :])

        res_list = []
        for item, x in zip(item_list, item_xs):
            if item not in self.items_pool.keys():                      # line 6
                self.items_pool[item] = (
                    FactorItem(self.lambda_2, self.d, self.l, x))       # line [7], 8
            xv_T_a, vec_0X0V_WT_a, CI_a = self.items_pool[item].calculation(user, self.WT, self.N)
            res_list.append(
                xv_T_a.dot(self.mat_theta_Wu)
                + self.alpha_u * np.sqrt(
                    vec_0X0V_WT_a.dot(self.AI).dot(vec_0X0V_WT_a.T))
                + self.alpha_a * np.sqrt(self.mat_theta_Wu.dot(CI_a).dot(self.mat_theta_Wu.T))
            )                                                           # line 9
        self.rec_item = item_list[np.argmax(res_list)]
        return self.rec_item

    def update(self, r):                                                # line 10
        vec_0X0V_WT_a = self.items_pool[self.rec_item].vec_0X0V_WT_a
        self.A += vec_0X0V_WT_a.dot(vec_0X0V_WT_a.T)                    # line 11
        self.b += vec_0X0V_WT_a * r                                     # line 12

        self.AI = np.linalg.inv(self.A)
        self.theta = self.AI.dot(self.b)                                # line 13
        self.items_pool[self.rec_item]\
            .update(self.mat_theta_Wu[:self.d, :],
                    self.mat_theta_Wu[self.d:, :], r)                   # line [14 - 17]
