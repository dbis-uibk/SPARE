import pickle
import argparse
from math import sqrt
from tqdm import tqdm
import numpy as np
import random
import itertools
from collections import Counter, OrderedDict

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

import os
import util.istarmap
from multiprocessing import Pool, Manager


def process_session(sess_id, sess, all_pos_idx, all_neg_idx):
    # pos sessions
    # get sessions with same target item
    pos_idx = [n_sess_id for n_sess_id in target_session_id_map[targets[sess_id]] if n_sess_id != sess_id]

    # no pos session -> use anchor session
    if len(pos_idx) == 0:
        all_pos_idx[sess_id] = [sess_id]
        pos_sessions = []
    elif len(pos_idx) < sample_num:
        all_pos_idx[sess_id] = pos_idx
        pos_sessions = [sessions[pos_id] for pos_id in pos_idx]
    else:
        sampled_pos_idx = random.sample(pos_idx, sample_num)
        pos_sessions = [sessions[pos_id] for pos_id in sampled_pos_idx]
        all_pos_idx[sess_id] = sampled_pos_idx

    # hard negatives: same items, but different target
    candidate_neg_idx = []
    sess_item_idx = []
    for item_id in sess:
        sess_item_idx = sess_item_idx + list(item_session_id_map.get(item_id))

    candidate_neg_idx = candidate_neg_idx + [idx for idx in sess_item_idx if targets[idx] != targets[sess_id]]

    # reduce complexity with sampling random neg sessions
    if len(candidate_neg_idx) > sample_num * 8:
        candidate_neg_idx = random.sample(candidate_neg_idx, sample_num * 8)

    # kick out subsets candidates (mostly from augmentation)
    candidate_neg_idx = [idx for idx in candidate_neg_idx if not set(sessions[idx]).issubset(set(sess))]

    if sample_num - len(candidate_neg_idx) > 0:
        candidate_neg_idx = candidate_neg_idx + \
                            random.sample(list(set([x for x in range(0, len(sessions))]) - set(pos_idx + [sess_id])),
                                          sample_num - len(candidate_neg_idx))
        all_neg_idx[sess_id] = candidate_neg_idx
    else:
        sim = [bleu_score.sentence_bleu([sess] + pos_sessions,
                                        sessions[idx],
                                        smoothing_function=SmoothingFunction().method7,
                                        weights=[0.5, 0.3, 0.15, 0.05])
               for idx in candidate_neg_idx]
        max_ind = np.argpartition(sim, -sample_num)[-sample_num:]
        all_neg_idx[sess_id] = [candidate_neg_idx[i] for i in max_ind]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='diginetica/tmall/nowplaying')
parser.add_argument('--sample-num', type=int, default=8)
opt = parser.parse_args()
print(opt)

dataset = opt.dataset
sample_num = opt.sample_num

seqs = pickle.load(open('datasets/' + dataset + '/train.pkl', 'rb'))
sessions = seqs[0]
targets = seqs[1]

item_session_id_map = {}
target_session_id_map = {}

for sess_idx, sess in enumerate(sessions):
    for item_id in sess:
        if item_id in item_session_id_map:
            item_session_id_map[item_id].add(sess_idx)
        else:
            item_session_id_map[item_id] = {sess_idx}

for sess_idx, target in enumerate(targets):
    if target in target_session_id_map:
        target_session_id_map[target].append(sess_idx)
    else:
        target_session_id_map[target] = [sess_idx]

if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    print(os.cpu_count())
    manager = Manager()
    shared_pos_dict = manager.dict()
    shared_neg_dict = manager.dict()
    tasks = [(sess_id, sess, shared_pos_dict, shared_neg_dict) for sess_id, sess in enumerate(sessions)]

    for _ in tqdm(pool.istarmap(process_session, tasks), total=len(sessions), mininterval=10):
        pass

    pool.close()

    pos_list = list(OrderedDict(sorted(shared_pos_dict.items())).values())
    neg_list = list(OrderedDict(sorted(shared_neg_dict.items())).values())
    pickle.dump(pos_list, open('datasets/' + dataset + '/pos_idx.pkl', 'wb'))
    pickle.dump(neg_list, open('datasets/' + dataset + '/neg_idx.pkl', 'wb'))

    print(len(pos_list))
    print(len(neg_list))
    print("Similar sessions done.")
