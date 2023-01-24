import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLTask(nn.Module):

    def __init__(self, opt):
        super(SSLTask, self).__init__()
        self.opt = opt
        self.temperature = opt.temp
        self.K = opt.k

    def row_column_shuffle(self, x):
        shuffled_x = x[torch.randperm(x.size()[0])]
        shuffled_x = shuffled_x[:, torch.randperm(shuffled_x.size()[1])]
        return shuffled_x

    def score(self, x1, x2, dim=1):
        return torch.sum(torch.mul(x1, x2), dim=dim)  # cosine operation

    def forward(self, h_session, last_items_emb, pos_last_items_emb, neg_last_items_emb, pos_target_item_emb, neg_targets_item_emb, reduce=True):
        h_sess_k = h_session.unsqueeze(1).repeat(1, self.K, 1)
        anchor = F.normalize((h_session + last_items_emb).unsqueeze(1), p=2, dim=-1)

        pos1 = (h_session + pos_target_item_emb).unsqueeze(1)
        pos2 = h_sess_k + pos_last_items_emb
        pos = torch.concat((pos1, pos2), dim=1)

        neg1 = h_sess_k + neg_targets_item_emb
        neg2 = h_sess_k + neg_last_items_emb
        neg = torch.concat((neg1, neg2), dim=1)

        pos = F.normalize(pos, p=2, dim=-1)
        neg = F.normalize(neg, p=2, dim=-1)

        pos_score = self.score(anchor.repeat(1, self.K + 1, 1), pos, dim=2)
        neg_score = self.score(anchor.repeat(1, self.K * 2, 1), neg, dim=2)

        pos_score = torch.sum(torch.exp(pos_score / self.temperature), 1)
        neg_score = torch.sum(torch.exp(neg_score / self.temperature), 1)

        if reduce:
            loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        else:
            loss = -torch.log(pos_score / (pos_score + neg_score))

        return loss