# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
from copy import deepcopy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.utils.svrg_cl import SVRG_CL


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--svrg', type=bool, default=False, help='Use svrg_cl or not.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.Tot_grad = SVRG_CL(self.loss, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()

        if self.args.svrg:
            if not self.buffer.is_empty():
                # 利用Memory和history_grad按照SVRG的思想进行一步梯度下降
                buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size)
                self.Tot_grad.update_and_replay(self.net, self.lastnet, buf_inputs, buf_labels)

            self.lastnet = deepcopy(self.net)

        self.opt.step()

        return loss.item()
    
    def end_task(self, dataset):
        self.net.eval()
        self.net.zero_grad()
        for i, data in enumerate(dataset.train_loader):
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            not_aug_inputs = not_aug_inputs.to(self.device)
            outputs = self.net(inputs)
            self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
            
        if self.args.svrg:    
            self.Tot_grad.update_history(self.net, dataset.train_loader)