import DataSetLink as DLSet
from tools import *
import random
import Constant
from A_DataPreprocess.BanditData import BanditData
import sys; sys.path.append('../')


def load_basic_data():
    logs = load_data(DLSet.logs_link)
    sub_logs = load_data(DLSet.sub_logs_link)
    user_context = load_obj(DLSet.user_context_link)
    item_context = load_obj(DLSet.item_context_link)
    user_selected = load_obj(DLSet.user_selected_link)
    return logs, sub_logs, user_context, item_context, user_selected


def gen_data(diff_logs, data_count, user_context, item_context, user_selected):
    result_logs = []
    diff_logs = diff_logs.sort_values(['timestamp'], ascending=True)

    # generate bandit data
    items_pool = set([i for i in range(data_count['itemID'])])
    cnt = 0
    for u, i, t, time in diff_logs.values:
        arm_context = {}
        bandit_context = {}
        arm_true_reward = {}
        rewards = {}

        # set info and sample arm_set
        bandit_context['context'] = list(user_context[u])
        bandit_context['tags'] = t
        bandit_context['tag_reward'] = 1
        arm_set = list(random.sample(items_pool - user_selected[u], Constant.n_arm_set - 1))
        arm_set.append(i)

        # load the item context and set the reward
        for item in arm_set:
            # arm_context[item] = list(item_context[item])
            arm_true_reward[item] = 1 if item == i else 0
            rewards[item] = 1 if item == i else 0

        # format as banditData
        bandit_data = BanditData(timestamp=t, arm_reward=rewards, arm_context=arm_context,
                                 arm_true_reward=arm_true_reward, bandit_id=u, bandit_context=bandit_context)
        result_logs.append(str(bandit_data.__dict__) + '\n')

        cnt += 1
        # print("%d / %d" % (cnt, diff_logs.shape[0]))
    return result_logs


def gen_social_matrix(social, map_user, data_count):
    social = replace(social, map_user, 'A', 'userID')
    social = replace(social, map_user, 'B', 'userID')

    social_mat = np.zeros((data_count['userID'], data_count['userID']))
    for each in social.values:
        if each[0] < data_count['userID'] and each[1] < data_count['userID']:
            social_mat[each[0], each[1]] = 1
            social_mat[each[1], each[0]] = 1
    social_mat = social_mat / (social_mat.sum(axis=0) + 0.0000001)
    return social_mat


def main():
    # load basic data
    logs, sub_logs, user_context, item_context, user_selected = load_basic_data()

    # calculate the diff_logs and data_count
    diff_logs = gen_diff_set(logs, sub_logs)
    data_count = cal_data_count(logs)

    # generate the bandit data
    result_logs = gen_data(diff_logs, data_count, user_context, item_context, user_selected)
    batch_write(result_logs, DLSet.bandit_data_link)

    # generate the relation matrix
    social = load_data(DLSet.social_filename, '\t')
    map_user = load_data(DLSet.map_link % 'userID')
    social_mat = gen_social_matrix(social, map_user, data_count)
    store_obj(social_mat, DLSet.social_mat_link)


if __name__ == '__main__':
    main()
