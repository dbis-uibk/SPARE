from functools import reduce
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from model.layers import *
from model.losses import *


class GraphRecommender(nn.Module):
    def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
        super(GraphRecommender, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.len_session = len_session

        self.dim = opt.dim

        self.item_embedding = nn.Embedding(num_node + 1, self.dim,
                                           padding_idx=0)
        self.pos_embedding = nn.Embedding(self.len_session, self.dim)

        self.ssl_task = SSLTask(opt)

        self.item_conv = GlobalItemConv(layers=opt.layers)
        self.w_k = opt.w_k
        self.adj = adj
        self.dropout = opt.dropout

        self.n_sessions = n_train_sessions
        self.memory_bank = torch.empty((n_train_sessions, self.dim))

        # pos attention
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        mask = torch.unsqueeze((item_seq != 0), -1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = hidden

        if rev_pos:
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = torch.flip(pos_emb, [0])  # reverse order
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        if attn:
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            sess_emb = torch.sum(beta * hidden, 1)
        else:
            sess_emb = torch.sum(nh * hidden, 1)

        return sess_emb

    def compute_con_loss(self, batch, sess_emb, item_embs):
        mask = torch.unsqueeze((batch['inputs'] != 0), -1)
        last_item_pos = torch.sum(mask, dim=1) - 1
        last_items = torch.gather(batch['inputs'], dim=1, index=last_item_pos).squeeze()
        last_items_emb = item_embs[last_items]

        pos_last_items_emb = item_embs[batch['pos_last_items']]
        neg_last_items_emb = item_embs[batch['neg_last_items']]

        pos_target_item_emb = item_embs[batch['targets']]
        neg_targets_item_emb = item_embs[batch['neg_targets']]

        con_loss = self.ssl_task(sess_emb, last_items_emb, pos_last_items_emb, neg_last_items_emb,
                                       pos_target_item_emb, neg_targets_item_emb)

        return con_loss


    def forward(self, batch, cl=False):
        items, inputs, alias_inputs = batch['items'], batch['inputs'], batch['alias_inputs']
        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
        hidden = graph_item_embs[items]

        # dropout
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        # reverse position attention
        sess_emb = self.compute_sess_emb(inputs, seq_hidden, rev_pos=True, attn=True)

        # weighted L2 normalization: NISER, DSAN, STAN, COTREC
        select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
        graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)

        scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))

        con_loss = torch.Tensor(0)
        if cl:
            con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)

        return scores, con_loss
