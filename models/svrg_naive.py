# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from copy import deepcopy
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'naive SVRG.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class SVRGNAIVE(ContinualModel):
    NAME = 'svrg_naive'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SVRGNAIVE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.Tot_grad = None
        self.old_model = None
        self.epoch_cnt = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if self.Tot_grad is not None:
            idx = 0
            old_grad = 0.005 * (self.Tot_grad - self.cac_grad(self.old_model, buf_inputs, buf_labels))
            for param in self.net.parameters():
                param.grad += old_grad[idx].to(self.device)
                idx += 1

        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        self.net.eval()
        self.net.zero_grad()
        for i, data in enumerate(dataset.train_loader):
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            not_aug_inputs = not_aug_inputs.to(self.device)
            self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)
            
    def epoch_task(self):
        if not self.buffer.is_empty():
            self.epoch_cnt += 1
            if self.epoch_cnt % 5 == 0:
                inputs, labels = self.buffer.get_all_data()
                self.Tot_grad = self.cac_grad(self.net, inputs, labels)
                self.old_model = deepcopy(self.net)

    def cac_grad(self, net, inputs, labels):
        model = deepcopy(net)
        model.eval()
        model.zero_grad()
        outputs = model(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        ans = np.array([x.grad.cpu() for x in model.parameters()])
        return ans