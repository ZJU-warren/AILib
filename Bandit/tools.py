import pandas as pd
import numpy as np
import pickle
import json
import os


# generate org_df - sub_df
def gen_diff_set(org_df, sub_df):
    df = pd.concat([org_df, sub_df])
    return df.drop_duplicates(df.columns, keep=False)


# calculate the data count
def cal_data_count(df, index_start=0):
    return (df.max() + 1 - index_start).to_dict()


# load data as table
def load_data(data_link, sep=',', header='infer', names=None):
    return pd.read_csv(open(data_link, 'r'), sep=sep, header=header, names=names)


# store object
def store_obj(obj, data_link):
    pickle.dump(obj, open(data_link, 'wb'))


# load object
def load_obj(data_link):
    return pickle.load(open(data_link, 'rb'))


# replace the keyword in org_df with its map id
def replace(org_df, map_df, l_name, r_name):
    cols = org_df.columns
    org_df = pd.merge(org_df, map_df, left_on=[l_name], right_on=[r_name])
    org_df[l_name] = org_df[r_name + '_MapID']
    org_df = org_df[cols]
    return org_df


# map the keyword field to [0, .., N)
def map_id(org_df, keyword, map_link=None):
    df = org_df.drop_duplicates([keyword], keep='last')[[keyword]]

    # generate the map-table and store
    df['tempMark'] = 1
    df[keyword + '_MapID'] = df.groupby(['tempMark']).cumcount()
    df = df.drop(columns=['tempMark'])

    # replace the keyword in org_df with its map id
    org_df = replace(org_df, df, keyword, keyword)

    # store the map table
    if map_link is not None:
        df.to_csv(map_link, index=False)

    return org_df


# write logs by batch
def batch_write(logs, data_link, batch_size=100000):
    batch_count = len(logs) // batch_size + (1 if len(logs) % batch_size != 0 else 0)
    total = len(logs)

    for i in range(0, batch_count):
        with open(data_link % (i+1), 'w') as f_bandit_data:
            start = i * batch_size
            for each in logs[start: min(start + batch_size, total)]:
                f_bandit_data.write(each + '\n')



