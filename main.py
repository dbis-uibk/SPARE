import argparse

import scipy.sparse

from util.util import *
import pickle
import csv

from model.recommender import *
from model.sampler import *
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/diginetica/nowplaying/tmall/retailrocket')
parser.add_argument('--len-session', type=int, default=50, help='maximal session length')

parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--spare', type=int, default=1)
parser.add_argument('--layers', type=int, default=1, help='the number of gnn layers')
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--cl-base', type=int, default=0, help='Use contrastive base loss with pos and neg sessions')
parser.add_argument('--beta', type=float, default=0.1, help='weighting contrastive loss')
parser.add_argument('--margin', type=float, default=1.0, help='triplet margin')
parser.add_argument('--w-k', type=int, default=12, help='weight l2 normalization, ~10-20')

parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--log-interval', type=int, default=500, help='print the loss after this number of iterations')
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--seed', type=int, default=None)

opt = parser.parse_args()

def main():
    seed = init_seed(opt.seed)

    if opt.dataset == 'nowplaying':
        num_node = 60416
        opt.w_k = 11
        opt.dropout = 0.0
        opt.beta = 0.01
        opt.margin = 0.5
    elif opt.dataset == 'retailrocket':
        num_node = 36968
        opt.w_k = 12
        opt.dropout = 0.2
        opt.beta = 0.1
        opt.margin = 1.0
    elif opt.dataset == 'tmall':
        num_node = 40727
        opt.w_k = 16
        opt.dropout = 0.4
        opt.beta = 0.01
        opt.margin = 0.25
    else:
        num_node = 309

    print(opt)
    print('reading dataset')
    
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.pkl', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.pkl', 'rb'))

    pos_idx, neg_idx = None, None
    if opt.cl_base:
        pos_idx = pickle.load(open('datasets/' + opt.dataset + '/pos_idx.pkl', 'rb'))
        neg_idx = pickle.load(open('datasets/' + opt.dataset + '/neg_idx.pkl', 'rb'))

    global_adj_coo = scipy.sparse.load_npz('datasets/' + opt.dataset + '/adj_global.npz')
    sparse_global_adj = trans_to_cuda(sparse2sparse(global_adj_coo))

    train_data = Data(opt, train_data, pos_idx, neg_idx, global_adj_coo.tocsr(), opt.len_session, num_node, train=True)
    test_data = Data(opt, test_data, None, None, global_adj_coo.tocsr(), opt.len_session, num_node, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                              shuffle=False, pin_memory=True)
    model = trans_to_cuda(GraphRecommender(opt, num_node, sparse_global_adj, len_session=train_data.max_len,
                                              n_train_sessions=len(train_data)))
    print(model)

    trainer = Trainer(
        model,
        None,#sampler,
        train_loader,
        test_loader,
        opt=opt,
        Ks=[5, 10, 20],
    )

    print('start training')
    best_results = trainer.train(opt.epochs, opt.log_interval)

if __name__ == '__main__':
    main()
