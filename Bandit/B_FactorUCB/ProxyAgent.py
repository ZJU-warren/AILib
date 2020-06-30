import sys; sys.path.append('../')
from tools import *
from B_FactorUCB.FactorUCB import FactorUCB
import DataSetLink as DLSet
import Constant


class EvalAgent:
    pass


class ProxyAgent:
    def __init__(self, d, l, N, W, lambda_1, lambda_2, alpha_u, alpha_a, item_context):
        self.model = FactorUCB(d, l, N, W, lambda_1, lambda_2, alpha_u, alpha_a)
        self.item_context = item_context
        self.item_pool = set()

    def one_step(self, data):
        item_list = []
        item_xs = []
        for arm in data['arm_reward'].keys():
            item_list.append(arm)
            if arm not in self.item_pool:
                item_xs.append(self.item_context[arm])
            else:
                item_xs.append('_')
        rec = self.model.decide(item_list, item_xs, data['bandit_id'])
        # print('rec is', rec)
        if data['arm_reward'][rec] != 0:
            print('hit')
        self.model.update(data['arm_reward'][rec])


def main():
    W = load_obj(DLSet.social_mat_link)
    item_context = load_obj(DLSet.item_context_link)

    agent = ProxyAgent(
        d=Constant.pca_component, l=5, N=W.shape[0], W=W,
        lambda_1=1, lambda_2=1, alpha_u=0.1, alpha_a=0.1,
        item_context=item_context
    )

    T = 1
    for i in range(1, T + 1):
        with open(DLSet.bandit_data_link % 1, 'r') as f:
            for log in f:
                if log != '\n':
                    agent.one_step(eval(log))


if __name__ == '__main__':
    main()


