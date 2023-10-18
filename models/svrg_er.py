# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
from copy import deepcopy
import torch
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' SVRG + ER.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--gamma', type=float, required=True,
                        help='Total grad weight.')
    parser.add_argument('--replay_every', type=int, required=True,
                        help='Replay every xx batch.')
    parser.add_argument('--start_svrg_ep', type=int, required=True,
                        help='Start using SVRG grad from epoch xx.')
    return parser

class ToT_grad:
    def __init__(self):
        self.count = 0
        self.history_grad = None
        self.record = None
        self.best = None

    def update_history(self, net, train_loader, device):
        pass

    def update_record(self, g):
        self.count += 1
        if self.record is None:
            self.record = deepcopy(g)
        else:
            self.record += g

    def update_epoch(self):
        if self.record is not None:
            if self.history_grad is not None:
                self.history_grad += self.record / self.count
            else:
                self.history_grad = self.record / self.count
            self.count = 0
            self.record = None

    def update_best(self):
        self.best = deepcopy(self.history_grad)

    def recover_from_best(self):
        self.history_grad = self.best

    def get_history(self):
        return self.history_grad


class SVRGER(ContinualModel):
    NAME = 'svrg_er'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(SVRGER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.Tot_grad = ToT_grad()

    def observe(self, inputs, labels, not_aug_inputs, idx, epoch, cur_task):

        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if not self.buffer.is_empty() and idx % self.args.replay_every == self.args.replay_every - 1:
            # 利用Memory和history_grad按照SVRG的思想进行一步梯度下降
            now = 0
            cur_grad = self.cac_grad(self.net)

            last_grad = self.cac_grad(self.lastnet)

            self.Tot_grad.update_record(cur_grad - last_grad)

            if epoch > self.args.start_svrg_ep:
                tot = self.Tot_grad.get_history() * self.args.gamma / cur_task
                for param in self.net.parameters():
                    param.grad += tot[now].to(self.device)
                    now += 1

        self.lastnet = deepcopy(self.net)
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        # self.Tot_grad.update_history()
        pass

    def epoch_task(self):
        self.Tot_grad.update_epoch()
    
    def cac_grad(self, model):
        model = deepcopy(model)
        model.train()
        model.zero_grad()
        buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
        buf_outputs = model(buf_inputs)
        loss = self.loss(buf_outputs, buf_labels)
        loss.backward()

        ans = np.array([x.grad.cpu() for x in model.parameters()])
        return ans