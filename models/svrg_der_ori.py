# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
from copy import deepcopy
import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
from models.utils.svrg_cl import SVRG_CL


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' SVRG + Dark Experience Replay++ (Original way).')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--svrg', type=bool, required=True,
                        help='Use svrg_cl or not.')
    return parser


class SVRGDerOri(ContinualModel):
    NAME = 'svrg_der_ori'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(SVRGDerOri, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.Tot_grad = SVRG_CL(self.loss, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        lossa = torch.tensor(100)
        lossb = torch.tensor(100)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            lossa = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += lossa

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            lossb = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss += lossb

        loss.backward()

        if not self.buffer.is_empty():
            # 利用Memory和history_grad按照SVRG的思想进行一步梯度下降
            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size * 4)
            self.Tot_grad.update_and_replay(self.net, self.lastnet, buf_inputs, buf_labels)

        self.lastnet = deepcopy(self.net)
        self.opt.step()
        return loss.item(), lossa.item(), lossb.item()

    def end_task(self, dataset):
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(not_aug_inputs)
                self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data)

        self.Tot_grad.update_history(self.net, dataset.train_loader)

        # if t == 4:
        #     self.args.alpha *= 2
        #     self.args.beta *= 2
        # elif t == 8:
        #     self.args.alpha *= 2
        #     self.args.beta *= 2
        # if t > 1 and t % 2 == 0:
        #     self.args.beta += 0.1
        #     self.args.alpha += 0.01
        #     self.weight_align(t)
    
    def weight_align(self, t):
        end = t * 10
        start = (t - 1) * 10
        weights = self.net.classifier.weight.data
        newnorm = torch.norm(weights[start:end, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:start, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.net.classifier.weight.data[start:end, :] *= gamma