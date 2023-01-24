import argparse

import scipy.sparse
import torch

from util import *
import pickle

from model.recommender import *
from trainer import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='lastfm/tmall/retailrocket')
parser.add_argument('--len-session', type=int, default=50, help='maximal session length')

parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--layers', type=int, default=1, help='the number of gnn layers')
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num-workers', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')

parser.add_argument('--cl', type=int, default=1, help='Use contrastive base loss with pos and neg sessions')
parser.add_argument('--k', type=int, default=4, help='Sample size for pos and neg contrastive samples')
parser.add_argument('--temp', type=float, default=0.2, help='Temperature parameter for CL')
parser.add_argument('--sim', type=str, default='bleu', help='Similarity measure for sessions: bleu/jaccard/cosine/levenshtein')
parser.add_argument('--beta', type=float, default=0.05, help='weighting contrastive loss')
parser.add_argument('--w-k', type=int, default=12, help='weight l2 normalization, ~10-20')

parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--log-interval', type=int, default=500, help='print the loss after this number of iterations')
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--seed', type=int, default=2022)

opt = parser.parse_args()


def main():
    seed = init_seed(opt.seed)
    print(seed)

    if opt.dataset == 'retailrocket':
        num_node = 36968
        opt.w_k = 12
        opt.dropout = 0.2
        opt.beta = 0.2
        opt.k = 16
        opt.temp = 0.2
    elif opt.dataset == 'tmall':
        num_node = 40727
        opt.w_k = 16
        opt.dropout = 0.4
        opt.beta = 0.2
        opt.k = 8
        opt.temp = 0.2
    elif opt.dataset == 'lastfm':
        num_node = 38615
        opt.w_k = 17
        opt.dropout = 0.4
        opt.beta = 0.05
        opt.k = 4
        opt.temp = 0.7

    print(opt)
    print('reading dataset')
    
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.pkl', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.pkl', 'rb'))

    global_adj_coo = scipy.sparse.load_npz('datasets/' + opt.dataset + '/adj_global.npz')
    sparse_global_adj = trans_to_cuda(sparse2sparse(global_adj_coo))

    train_data = DataSampler(opt, train_data, opt.len_session, num_node, train=True)
    test_data = DataSampler(opt, test_data, opt.len_session, num_node, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                              shuffle=False, pin_memory=False)
    model = trans_to_cuda(GraphRecommender(opt, num_node, sparse_global_adj, len_session=train_data.max_len,
                                              n_train_sessions=len(train_data)))
    print(model)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        opt=opt,
        Ks=[5, 10, 20],
    )

    print('start training')
    best_results = trainer.train(opt.epochs, opt.log_interval)

    print('\n')    
    print(opt)


if __name__ == '__main__':
    main()
