import time
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

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


def prepare_batch(batch, concat=False):
    if concat:
        batch_dict = {
                "index": trans_to_cuda(torch.concat((batch["index"], batch["pos_session"]["index"], batch["neg_session"]["index"]))).long(),
                "alias_inputs": trans_to_cuda(torch.concat((batch["alias_inputs"], batch["pos_session"]["alias_inputs"], batch["neg_session"]["alias_inputs"]))).long(),
                "items": trans_to_cuda(torch.concat((batch["items"], batch["pos_session"]["items"], batch["neg_session"]["items"]))).long(),
                "inputs": trans_to_cuda(torch.concat((batch["inputs"], batch["pos_session"]["inputs"], batch["neg_session"]["inputs"]))).long(),
                "targets": trans_to_cuda(batch["targets"]).long(),
            }
    else:
        batch_dict = {
            "alias_inputs": trans_to_cuda(batch["alias_inputs"]).long(),
            "items": trans_to_cuda(batch["items"]).long(),
            "mask": trans_to_cuda(batch["mask"]).long(),
            "targets": trans_to_cuda(batch["targets"]).long(),
            "inputs": trans_to_cuda(batch["inputs"]).long(),
            "index": trans_to_cuda(batch["index"]).long(),
            "neighbors": trans_to_cuda(batch["neighbors"]).long(),
        }

    return batch_dict


def evaluate(model, data_loader, Ks=[10, 20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with torch.no_grad():
        for batch in data_loader:
            batch_dict = prepare_batch(batch)

            select, scores, _ = model(batch_dict['items'],
                                      batch_dict['inputs'],
                                      batch_dict['alias_inputs'])

            loss = F.cross_entropy(scores, batch_dict['targets'])
            results['Loss'] -= loss.item()

            batch_size = scores.size(0)
            num_samples += batch_size
            topk = torch.topk(scores, k=max_K, sorted=True)[1]
            targets = batch_dict['targets'].unsqueeze(-1)
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
            sampler,
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
        self.contrastive_base = opt.cl_base
        self.contrastive_rl = opt.cl_rl
        self.beta = opt.beta
        self.gamma = opt.gamma
        self.margin = opt.margin

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        if sampler:
            self.sampler = sampler
            self.sampler_optimizer = torch.optim.Adam(self.sampler.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, epochs, log_interval=100):
        max_results = defaultdict(float)
        max_results['Loss'] = -np.inf
        max_epochs = defaultdict(int)
        bad_counter = 0
        t = time.time()
        con_loss, saved_rewards = torch.Tensor(0), torch.Tensor(0)
        total_loss, total_con_loss, total_reward, mean_loss, mean_con_loss, mean_reward = 0, 0, 0, 0, 0, 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                batch_size = batch["items"].size(0)
                batch = prepare_batch(batch, concat=self.contrastive_base and epoch >= 1)

                self.optimizer.zero_grad()

                h, scores, item_embeddings = self.model(batch['items'],
                                                            batch['inputs'],
                                                            batch['alias_inputs'])
                if self.contrastive_base:
                    h_anchor = h[:batch_size]
                    h_pos = h[batch_size:batch_size * 2]
                    h_neg = h[-batch_size:]
                    scores = scores[:batch_size]

                loss = self.loss_function(scores, batch['targets'])

                item_embeddings = item_embeddings.detach()
                if self.contrastive_base and epoch >= 1:
                        con_loss = self.model.ssl_task(h_anchor, h_pos, h_neg)

                        combined_loss = loss + self.beta * con_loss
                        combined_loss.backward()

                elif self.contrastive_rl:
                    neg_sessions, saved_log_probs, saved_rewards, h_neg, h_target = self.sampler.run_episode(batch,
                                                                                                             self.model,
                                                                                                             item_embeddings.detach(),
                                                                                                             h_anchor)

                    con_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='none')(h_anchor, h_target,
                                                                                       h_neg)  # TODO supervised triplet margin loss?
                    # con_loss = self.model.ssl_task2(self.model.item_embedding, h_pos, h_neg, last_items)
                    # con_loss = self.con_loss_function(h_pos_local, h_neg_global.detach(), d=1)
                    # con_loss = self.model.ssl_task(h_pos, h_target, h_neg, labels=batch_dict['targets'])

                    combined_loss = loss + self.beta * torch.mean(con_loss)
                    combined_loss.backward()

                    """train sampler"""
                    self.sampler_optimizer.zero_grad()

                    # subtract average reward from the policy gradient to reduce the variance (see KGpolicy paper)
                    # REINFORCE with baseline (mean)
                    reward_batch = saved_rewards - saved_rewards.mean()
                    # eps = np.finfo(np.float32).eps.item()
                    # reward_batch = (saved_rewards - saved_rewards.mean()) / (saved_rewards.std() + eps)

                    batch_size = reward_batch.size(1)
                    n = reward_batch.size(0) - 1
                    R = torch.zeros(batch_size, device=reward_batch.device)
                    returns = torch.zeros(reward_batch.size(), device=reward_batch.device)

                    for i, r in enumerate(reward_batch.flip(0)):
                        R = r + self.gamma * R
                        returns[n - i] = R

                    reinforce_loss = -1 * torch.sum(returns * saved_log_probs)
                    reinforce_loss.backward()
                    self.sampler_optimizer.step()
                else:
                    loss.backward()

                self.optimizer.step()

                if log_interval:
                    mean_loss += loss.item() / log_interval
                    mean_con_loss += torch.mean(con_loss).item() / log_interval
                    mean_reward += saved_rewards.mean() / log_interval

                total_loss += loss.item()
                total_con_loss += torch.mean(con_loss).item()
                total_reward += saved_rewards.mean()

                if log_interval and self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Con-Loss = {mean_con_loss:.4f}, Reward = {mean_reward:.2f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss, mean_con_loss, mean_reward = 0, 0, 0
                self.batch += 1

            curr_results = evaluate(
                self.model, self.test_loader, Ks=self.Ks
            )

            '''# hpt
            nni.report_intermediate_result(round(curr_results['MRR@20'] * 100, 2))'''

            if log_interval:
                print(f'\nEpoch {self.epoch}:')
                print('Loss:\t%.3f' % total_loss)
                print('Con-Loss:\t%.3f' % total_con_loss)
                print('Total-Reward:\t%.3f' % total_reward)
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
            total_reward = 0.0

        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results
