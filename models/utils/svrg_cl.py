from copy import deepcopy
import torch
import numpy as np

class SVRG_CL:
    def __init__(self, loss, device, update_every=5, gamma=5e-6):
        self.count = 0
        self.history_grad = None
        self.record = None
        self.best = None
        self.update_every = update_every
        self.gamma = gamma
        self.loss = loss
        self.device = device
        self.task_cnt = 0

    def update_and_replay(self, net, lastnet, inputs, labels):
        cur_grad = self._cac_grad(net, inputs, labels)
        last_grad = self._cac_grad(lastnet, inputs, labels)
        self.count += 1

        if self.record is None:
            self.record = cur_grad - last_grad
        else:
            self.record += (cur_grad - last_grad)

        if self.count == self.update_every:
            if self.history_grad is not None:
                self.history_grad += self.record / self.count / self.task_cnt
            else:
                self.history_grad = self.record / self.count / self.task_cnt
            self.count = 0
            self.record = None

        if self.history_grad is not None:
            idx = 0
            tot = self.history_grad * self.gamma
            for param in net.parameters():
                param.grad += tot[idx].to(self.device)
                idx += 1
        
    def update_history(self, net, loader):
        self.task_cnt += 1
        status = net.training
        net.eval()
        net.zero_grad()
        for i, data in enumerate(loader):
            inputs, labels, not_aug_inputs = data
            not_aug_inputs, labels = not_aug_inputs.to(self.device), labels.to(self.device)
            outputs = net(not_aug_inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
        
        curgrad = np.array([x.grad.cpu() for x in net.parameters()])    
        if self.history_grad is None:
            self.history_grad = curgrad / len(loader) / self.task_cnt
        else:
            self.history_grad *= (self.task_cnt - 1) / self.task_cnt
            self.history_grad += curgrad / len(loader) / self.task_cnt
        net.train(status)

    def update_history_batch(self, net, inputs, labels):
        self.task_cnt += 1
        status = net.training
        net.eval()
        net.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        
        curgrad = np.array([x.grad.cpu() for x in net.parameters()])    
        if self.history_grad is None:
            self.history_grad = curgrad / self.task_cnt
        else:
            self.history_grad *= (self.task_cnt - 1) / self.task_cnt
            self.history_grad += curgrad / self.task_cnt
        net.train(status)

    def _cac_grad(self, net, inputs, labels):
        model = deepcopy(net)
        model.eval()
        model.zero_grad()
        outputs = model(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        ans = np.array([x.grad.cpu() for x in model.parameters()])
        return ans

    def update_best(self):
        self.best = deepcopy(self.history_grad)

    def recover_from_best(self):
        self.history_grad = self.best

    def get_history(self):
        return self.history_grad