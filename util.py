import numpy as np
import time
import datetime
from tqdm import tqdm
import pandas as pd
import itertools
import random
import copy
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

from model.distances import jaccard, tanimoto, cosine, dameraulevenshtein


def sparse2sparse(coo_matrix):
    v1 = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = coo_matrix.shape
    sparse_matrix = torch.sparse.LongTensor(i, v, torch.Size(shape))
    return sparse_matrix


def dense2sparse(matrix):
    a_ = scipy.sparse.coo_matrix(matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix


def handle_data(inputs, train_len=None):
    len_data = [len(nowData) for nowData in inputs]
    if train_len is None:
        max_len = max(len_data) + 1
    elif max(len_data) < train_len:
        max_len = max(len_data)
    else:
        max_len = train_len + 1
    us_pois = [upois + [0] * (max_len - le) if le < max_len else upois[-max_len:]
               for upois, le in zip(inputs, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def init_seed(seed=None):
    if seed is None or seed == 0:
        seed = int(time.time() * 1000 // 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    return seed
        

class DataSampler(Dataset):
    def __init__(self, opt, sessions, max_len, num_node, train=True):
        self.opt = opt
        inputs, mask, len_max = handle_data(sessions[0], max_len)
        self.sessions = sessions[0]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(sessions[1])
        self.mask = np.asarray(mask)
        self.length = len(sessions[0])
        self.max_len = len_max
        self.num_node = num_node
        self.vn_id = num_node + 1
        self.train = train

        self.k = opt.k
        self.num_subsample = self.k * 2
        self.similarity = opt.sim

        self.item_session_id_map = {}
        self.target_session_id_map = {}

        for sess_idx, sess in enumerate(sessions[0]):
            for item_id in sess:
                if item_id in self.item_session_id_map:
                    self.item_session_id_map[item_id].add(sess_idx)
                else:
                    self.item_session_id_map[item_id] = {sess_idx}

        for sess_idx, target in enumerate(sessions[1]):
            if target in self.target_session_id_map:
                self.target_session_id_map[target].append(sess_idx)
            else:
                self.target_session_id_map[target] = [sess_idx]

    def get_data(self, idx):
        u_input, mask, target = self.inputs[idx], self.mask[idx], self.targets[idx]

        node = np.unique(u_input)
        items = node.tolist() + (self.max_len - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return u_input, mask, target, items, alias_inputs

    def get_pos_last_items(self, session_id):
        target = self.targets[session_id]

        # sessions with same target item = positive sessions
        target_sessions_ids = copy.deepcopy(self.target_session_id_map[target])
        if len(target_sessions_ids) > 1:
            target_sessions_ids.remove(session_id)

        pos_ids = random.choices(target_sessions_ids, k=self.k)
        last_items = []
        for p_id in pos_ids:
            u_input, mask, target, items, alias_inputs = self.get_data(p_id)
            last_items.append(u_input[sum(mask) - 1])
        return last_items, pos_ids

    def session_similarity(self, session, pos_sessions, neg_sessions):
        sessions = [session] + pos_sessions
        if self.similarity == 'jaccard':
            sims = []
            for sess in sessions:
                sims.append([jaccard(set(sess), set(neg_sess)) for neg_sess in neg_sessions])
            return np.array(sims).sum(axis=0)
        elif self.similarity == 'tanimoto':
            sims = []
            for sess in sessions:
                sims.append([tanimoto(set(sess), set(neg_sess)) for neg_sess in neg_sessions])
            return np.array(sims).sum(axis=0)
        elif self.similarity == 'cosine':
            sims = []
            for sess in sessions:
                sims.append([cosine(set(sess), set(neg_sess)) for neg_sess in neg_sessions])
            return np.array(sims).sum(axis=0)
        elif self.similarity == 'levenshtein':
            sims = []
            for sess in sessions:
                sims.append([-dameraulevenshtein.damerau_levenshtein_distance(sess, neg_sess) for neg_sess in neg_sessions])
            return np.array(sims).sum(axis=0)
        elif self.similarity == 'bleu':
            return [bleu_score.sentence_bleu(sessions,
                                            neg_sess,
                                            smoothing_function=SmoothingFunction().method7,
                                            # weights=[0.25, 0.25, 0.25, 0.25] # TODO
                                            weights=[0.5, 0.3, 0.15, 0.05]
                                            )
                   for neg_sess in neg_sessions]

    def get_neg_sessions(self, session_id, pos_ids, pos_last_items):
        session = self.sessions[session_id]
        pos_sessions = [self.sessions[pos_id] for pos_id in pos_ids]
        pos_sessions = [list(x) for x in set(tuple(x) for x in pos_sessions)]  # uniqueness

        # hard negatives: same items, but different target
        candidate_neg_ids = []
        sess_item_idx = []
        for item_id in session:
            item_sessions = list(self.item_session_id_map.get(item_id))
            # reduce complexity lastfm
            if self.opt.dataset == 'lastfm' and len(item_sessions) > self.num_subsample:
                item_sessions = random.sample(item_sessions, self.num_subsample)

            sess_item_idx = sess_item_idx + item_sessions

        sess_item_idx = set(sess_item_idx)

        # reduce complexity lastfm
        if self.opt.dataset == 'lastfm' and len(sess_item_idx) > self.num_subsample:
            sess_item_idx = random.sample(sess_item_idx, self.num_subsample)

        candidate_neg_ids = candidate_neg_ids + [idx for idx in sess_item_idx if self.targets[idx] != self.targets[session_id]]

        # reduce complexity with sampling random neg sessions
        if len(candidate_neg_ids) > self.num_subsample:
            candidate_neg_ids = random.sample(candidate_neg_ids, self.num_subsample)

        # kick out subsets candidates (mostly from augmentation)
        candidate_neg_ids = [idx for idx in candidate_neg_ids if not set(self.sessions[idx]).issubset(set(session))]

        # kick out same session with same last items as pos & anchor
        candidate_neg_ids = [idx for idx in candidate_neg_ids if not self.sessions[idx][-1] in pos_last_items + [session[-1]]]

        if len(candidate_neg_ids) < self.k:
            # random fill up
            neg_ids = candidate_neg_ids + random.choices(range(0, len(self.sessions)), k=self.k - len(candidate_neg_ids))
        else:
            candidate_neg_sessions = [self.sessions[idx] for idx in candidate_neg_ids]
            sim = self.session_similarity(session, pos_sessions, candidate_neg_sessions)

            neg_ids = [candidate_neg_ids[idx] for idx in list(np.argsort(sim)[::-1])[:self.k]]

        last_items = []
        neg_targets = []
        for n_id in neg_ids:
            u_input, mask, target, items, alias_inputs = self.get_data(n_id)
            last_items.append(u_input[sum(mask) - 1])
            neg_targets.append(target)
        return last_items, neg_targets

    def __getitem__(self, index):
        u_input, mask, target, items, alias_inputs = self.get_data(index)

        pos_last_items, neg_last_items, neg_targets = None, None, None
        if self.train and self.opt.cl:
                pos_last_items, pos_ids = self.get_pos_last_items(index)
                neg_last_items, neg_targets = self.get_neg_sessions(index, pos_ids, pos_last_items)


        return {"alias_inputs": torch.tensor(alias_inputs),
                "items": torch.tensor(items),
                "mask": torch.tensor(mask),
                "targets": torch.tensor(target),
                "inputs": torch.tensor(u_input),
                "index": torch.tensor(index),
                "pos_last_items": torch.tensor(pos_last_items) if pos_last_items else torch.tensor([]),
                "neg_last_items": torch.tensor(neg_last_items) if neg_last_items else torch.tensor([]),
                "neg_targets": torch.tensor(neg_targets) if neg_targets else torch.tensor([]),
                }

    def __len__(self):
        return self.length
