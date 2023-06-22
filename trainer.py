import time
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

import numpy as np

from model.losses import *
from util import *


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def prepare_batch(batch):
    batch_dict = {
        "alias_inputs": trans_to_cuda(batch["alias_inputs"]).long(),
        "items": trans_to_cuda(batch["items"]).long(),
        "mask": trans_to_cuda(batch["mask"]).long(),
        "targets": trans_to_cuda(batch["targets"]).long(),
        "inputs": trans_to_cuda(batch["inputs"]).long(),
        "index": trans_to_cuda(batch["index"]).long(),
        "pos_last_items": trans_to_cuda(batch["pos_last_items"]).long(),
        "neg_last_items": trans_to_cuda(batch["neg_last_items"]).long(),
        "neg_targets": trans_to_cuda(batch["neg_targets"]).long(),
    }

    return batch_dict


def evaluate(model, data_loader, Ks=[10, 20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with torch.no_grad():
        for batch in data_loader:
            batch = prepare_batch(batch)

            scores, _ = model(batch)

            loss = F.cross_entropy(scores, batch['targets'])
            results['Loss'] -= loss.item()

            batch_size = scores.size(0)
            num_samples += batch_size
            topk = torch.topk(scores, k=max_K, sorted=True)[1]
            targets = batch['targets'].unsqueeze(-1)
            for K in Ks:
                hit_ranks = torch.where(topk[:, :K] == targets)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results


def print_results(results, epochs=None):
    print('Metric\t' + '\t'.join(results.keys()))
    print(
        'Value\t' +
        '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()])
    )
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            opt,
            Ks=[10, 20]
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.batch = 0
        self.patience = opt.patience
        self.Ks = Ks
        self.contrastive = opt.cl
        self.beta = opt.beta

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def train(self, epochs, log_interval=100):
        max_results = defaultdict(float)
        max_results['Loss'] = -np.inf
        max_epochs = defaultdict(int)
        bad_counter = 0
        t = time.time()
        total_loss, total_con_loss, mean_loss, mean_con_loss = 0, 0, 0, 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                batch = prepare_batch(batch)

                self.optimizer.zero_grad()

                scores, con_loss = self.model(batch, cl=self.contrastive)

                loss = self.loss_function(scores, batch['targets'])

                if self.contrastive:
                    combined_loss = loss + self.beta * con_loss
                    #combined_loss = con_loss
                    combined_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                if log_interval:
                    mean_loss += loss.item() / log_interval
                    mean_con_loss += torch.mean(con_loss).item() / log_interval

                total_loss += loss.item()
                total_con_loss += torch.mean(con_loss).item()

                if log_interval and self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Con-Loss = {mean_con_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss, mean_con_loss = 0, 0
                self.batch += 1

            curr_results = evaluate(
                self.model, self.test_loader, Ks=self.Ks
            )

            if log_interval:
                print(f'\nEpoch {self.epoch}:')
                print('Loss:\t%.3f' % total_loss)
                print('Con-Loss:\t%.3f' % total_con_loss)
                print_results(curr_results)

            any_better_result = False
            for metric in curr_results:
                if curr_results[metric] > max_results[metric]:
                    max_results[metric] = curr_results[metric]
                    max_epochs[metric] = self.epoch
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break

            self.scheduler.step()
            self.epoch += 1
            total_loss = 0.0
            total_con_loss = 0.0

        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results
