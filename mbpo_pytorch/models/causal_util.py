import torch
import torch.nn as nn
import math
import numpy as np

from mbpo_pytorch.models.ensemble_util import EnsembleLayer
from mbpo_pytorch.models.sparse_util import EnsembleSparseLayer
from mbpo_pytorch.thirdparty.trace_expm import trace_expm
from collections import OrderedDict

from typing import List, Dict, Union, Optional

from abc import ABC

class EnsembleCausalMLP(nn.Module):
    def __init__(self,
                 hidden_dims,
                 node_num,
                 activation='tanh',
                 last_activation='identity',
                 ensemble_size=7,
                 biases=None):
        super(EnsembleCausalMLP, self).__init__()
        self.hidden_dims = hidden_dims.copy() + [2]       # last layer is (mean, variance)
        biases = [True] * (len(self.hidden_dims) - 1) if biases is None else biases.copy()
        self.activation = getattr(torch, activation)
        # setattr in __init__.py of models
        self.last_activation = getattr(torch, last_activation)
        self.node_num = node_num
        self.ensemble_size = ensemble_size

        # fc1: variable splitting for l1
        self.fc1_pos = EnsembleLayer(node_num, node_num * self.hidden_dims[0],
                                     ensemble_size=ensemble_size, bias=biases[0])
        self.fc1_neg = EnsembleLayer(node_num, node_num * self.hidden_dims[0],
                                     ensemble_size=ensemble_size, bias=biases[0])
        # bound是对于fc1而言的，对于i==j，要求weight只能为0
        # 对于其他的参数，要求weight至少0，因此有两个weight，分别代表正和负
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        if 1 < len(self.hidden_dims):
            for l in range(len(self.hidden_dims) - 1):
                layers.append(EnsembleSparseLayer(node_num, self.hidden_dims[l], self.hidden_dims[l + 1],
                                                  ensemble_size=ensemble_size, bias=biases[l]))
        self.fc2 = nn.ModuleList(layers)

    def set_bound(self, bound):
        self.fc1_pos.weight.bounds = bound
        self.fc1_neg.weight.bounds = bound

    def _bounds(self):
        bounds = []
        for k in range(self.ensemble_size):
            for j in range(self.node_num):
                for m in range(self.hidden_dims[0]):
                    for i in range(self.node_num):
                        if i == j:
                            bound = (0, 0)
                        else:
                            bound = (0, None)
                        bounds.append(bound)
        return bounds

    def forward(self, x):
        # x size: batch-size, ensemble-size, node-num
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        # x size: batch-size, ensemble-size, (node-num * dims[0])
        x = x.view(-1, self.ensemble_size, self.node_num, self.hidden_dims[0])  # [n, d, m1]
        # x size: batch-size, ensemble-size, node-num, dims[0]
        for fc in self.fc2:
            x = self.activation(x)
            x = fc(x)
        # x size: batch-size, ensemble-size, node-num, 2
        x = self.last_activation(x)
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [e, j * m1, i]
        fc1_weight = fc1_weight.view(self.ensemble_size, d, -1, d)  # [e, j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=2)
        A = A.transpose(-1, -2)         # transpose
        h_list = [trace_expm(A[idx]) - d for idx in range(self.ensemble_size)]
        h = sum(h_list)     # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [e, j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [e, j * m1, i]
        fc1_weight = fc1_weight.view(self.ensemble_size, d, -1, d)  # [e, j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=2)
        A = A.transpose(-1, -2)         # transpose
        W = torch.sqrt(A)  # [e, i, j]
        W = W.cpu().detach().numpy()  # [e, i, j]
        return W


class EnsembleCausalModel(EnsembleCausalMLP):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int],
                 output_state_dim=None, ensemble_size=7):
        output_state_dim = output_state_dim or state_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_state_dim = output_state_dim
        self.reward_dim = reward_dim
        self.node_num = state_dim + action_dim + output_state_dim + reward_dim
        self.ensemble_size = ensemble_size
        super(EnsembleCausalModel, self).__init__(hidden_dims, self.node_num, ensemble_size=ensemble_size)

        self.load_model_bound()

    def load_model_bound(self):
        bounds = []
        for k in range(self.ensemble_size):
            for i in range(self.node_num):              # states + actions + states + reward(1)
                for m in range(self.hidden_dims[0]):
                    for j in range(self.node_num):
                        if i == j:
                            bound = (0, 0)
                        else:
                            bound = (0, None)
                        bounds.append(bound)
        return bounds