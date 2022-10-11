import numpy as np
import time
import datetime
from tqdm import tqdm
import pandas as pd
import itertools
import random
import scipy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction


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


def init_seed(seed=None, deterministic=False):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return seed


def process_session_data(inputs):
    get_uniques = lambda i: torch.unique(inputs[i], sorted=True,)
    get_aliases = lambda i: torch.unique(inputs[i], sorted=True, return_inverse=True)

    items = [get_uniques(i) for i in torch.arange(len(inputs))]
    items[0] = F.pad(items[0], (0, inputs.shape[1] - len(items[0])))
    items = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)

    alias_inputs = torch.stack([get_aliases(i)[1] for i in torch.arange(len(inputs))]).to(inputs.device)

    return items, alias_inputs


class Data(Dataset):
    def __init__(self, opt, sessions, pos_idx, neg_idx, global_adj, max_len, num_node, train=True):
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
        self.global_adj = global_adj
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.train = train

    def get_data(self, idx):
        u_input, mask, target = self.inputs[idx], self.mask[idx], self.targets[idx]

        node = np.unique(u_input)
        items = node.tolist() + (self.max_len - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return u_input, mask, target, items, alias_inputs

    def get_pos_session(self, anchor_id):
        pos_id = random.choice(self.pos_idx[anchor_id])
        u_input, mask, target, items, alias_inputs = self.get_data(pos_id)
        return {
            "index": pos_id,
            "alias_inputs": torch.tensor(alias_inputs),
            "items": torch.tensor(items),
            "inputs": torch.tensor(u_input),
        }

    def get_neg_session(self, anchor_id):
        neg_id = random.choice(self.neg_idx[anchor_id])
        u_input, mask, target, items, alias_inputs = self.get_data(neg_id)
        return {
            "index": neg_id,
            "alias_inputs": torch.tensor(alias_inputs),
            "items": torch.tensor(items),
            "inputs": torch.tensor(u_input),
        }


    def __getitem__(self, index):
        u_input, mask, target, items, alias_inputs = self.get_data(index)

        pos_session, neg_session = None, None
        if self.train:
            sampled_neighbors, rnd_negative_session = [], []

            if self.opt.cl_base:
                pos_session = self.get_pos_session(index)
                neg_session = self.get_neg_session(index)
        else:
            sampled_neighbors, rnd_target_session, rnd_negative_session = [], [], []


        return {"alias_inputs": torch.tensor(alias_inputs),
                "items": torch.tensor(items),
                "mask": torch.tensor(mask),
                "targets": torch.tensor(target),
                "inputs": torch.tensor(u_input),
                "index": torch.tensor(index),
                "neighbors": torch.tensor(sampled_neighbors),
                "pos_session": pos_session if pos_session else [],
                "neg_session": neg_session if neg_session else []
                }

    def __len__(self):
        return self.length
