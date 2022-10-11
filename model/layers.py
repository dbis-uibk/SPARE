import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalItemConv(nn.Module):
    def __init__(
            self,
            spare=True,
            layers=1,
            feat_drop=0.0
    ):
        super(GlobalItemConv, self).__init__()
        self.spare = spare
        self.layers = layers
        self.feat_drop = nn.Dropout(feat_drop)

    def forward(self, x, adj):
        h = x
        final = [x]
        for i in range(self.layers):
            h = torch.sparse.mm(adj, h)
            h = F.normalize(h, dim=-1, p=2)
            h = self.feat_drop(h)
            final.append(h)
        if self.layers > 1:
            h = torch.sum(torch.stack(final), dim=0) / (self.layers + 1)
        return h

    def __repr__(self):
        return '{}(n_layers={},dropout={})'.format(self.__class__.__name__, self.layers, self.feat_drop)