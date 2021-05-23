import torch
import torch.nn as nn
import math
import numpy as np

from mbpo_pytorch.models.ensemble_util import EnsembleLayer
from mbpo_pytorch.thirdparty.trace_expm import trace_expm
from collections import OrderedDict

from typing import List, Dict, Union, Optional

from abc import ABC


class EnsembleSparseLayer(nn.Module):
    def __init__(self, input_features, output_features, node_num, ensemble_size=7, bias=True):
        super(EnsembleSparseLayer, self).__init__()
        self.node_num = node_num
        self.ensemble_size = ensemble_size
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(ensemble_size,
                                                node_num,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, node_num, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # 初始化权重
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        # input-size: batch-size * ensemble-size * node-num * input-dim
        # weight: ensemble-size * node-num * input-dim * output-dim
        out = torch.einsum("abcd,bcde->abce", inputs, self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):
        return 'EnsembleCausalLayer(ensemble_size={}, input_features={}, output_features={}, node_num={}, bias={})'.\
            format(self.ensemble_size, self.input_features, self.output_features, self.node_num,
                   self.bias is not None)

    def __str__(self):
        return 'ensemble_size={}, input_features={}, output_features={}, self.node_num, bias={}'.format(
            self.ensemble_size, self.input_features, self.output_features, self.node_num,
            self.bias is not None)


class EnsembleSparseMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 sparse_num,
                 hidden_dims,
                 activation='tanh',
                 last_activation='identity',
                 ensemble_size=7,
                 biases=None):
        super(EnsembleSparseMLP, self).__init__()
        self.hidden_dims = [input_dim] + hidden_dims.copy() + [output_dim]

        biases = [True] * (len(self.hidden_dims) - 1) if biases is None else biases.copy()
        biases[0] = False

        self.activation = getattr(torch, activation)
        # setattr in __init__.py of models
        self.last_activation = getattr(torch, last_activation)
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.sparse_num = sparse_num

        layers = []
        for l in range(len(self.hidden_dims) - 1):
            layers.append(EnsembleSparseLayer(self.hidden_dims[l], self.hidden_dims[l + 1], node_num=sparse_num,
                                              ensemble_size=ensemble_size, bias=biases[l]))
        self.fc = nn.ModuleList(layers)


    def forward(self, x):
        # x size: batch-size, ensemble-size, input-dim
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, 1, self.sparse_num, 1)
        # x size: batch-size, ensemble-size, output-dim, input-dim
        for fc in self.fc:
            x = self.activation(x)
            x = fc(x)
        # x size: batch-size, ensemble-size, node-num, 2
        x = self.last_activation(x)
        return x

    def compute_l1_loss(self):
        """Take l1 norm of fc1 weight"""
        # shape: ensemble-size, output-dim, input-dim, hidden-dim[0]
        weight = self.fc[0].weight
        weight = torch.abs(weight)
        l1_obj, _ = torch.max(weight, dim=-1)
        l1_obj = torch.sum(l1_obj)
        return l1_obj

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
        weight_norms = []
        for name, weight in self.named_parameters():
            if "weight" in name:
                weight_norms.append(weight.norm(2))
        weight_norms = torch.stack(weight_norms, dim=0)
        weight_decay = (torch.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay


class EnsembleSparseModel(EnsembleSparseMLP):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int],
                 output_state_dim=None, ensemble_size=7):
        self.input_dim = state_dim + action_dim
        output_state_dim = output_state_dim or state_dim
        self.output_dim = output_state_dim + reward_dim

        self.output_state_dim = output_state_dim
        self.ensemble_size = ensemble_size
        super(EnsembleSparseModel, self).__init__(self.input_dim, 2, self.output_dim, hidden_dims,
                                                  ensemble_size=ensemble_size)

    def forward(self, states, actions):
        # input size: batch-size * ensemble-size * dim
        inputs = torch.cat([states, actions], dim=-1)
        # print(inputs.shape)
        outputs = super(EnsembleSparseModel, self).forward(inputs)
        batch_size = outputs.shape[0]
        outputs = outputs.contiguous().view(batch_size, self.ensemble_size, 2 * self.output_dim)
        outputs = [{'diff_states': outputs[:, i, :self.output_state_dim * 2],
              'rewards': outputs[:, i, self.output_state_dim * 2:]}
             for i in range(self.ensemble_size)]
        return outputs

    def get_state_dict(self, index):
        assert index < self.ensemble_size

        ensemble_state_dict = self.state_dict()
        single_state_dict = OrderedDict()
        for i, key in enumerate(ensemble_state_dict):
            single_state_dict[key] = ensemble_state_dict[key][index].clone()

        return single_state_dict

    def set_state_dict(self, index, single_state_dict):
        assert index < self.ensemble_size

        new_state_dict = OrderedDict()
        ensemble_state_dict = self.state_dict()
        for i, key in enumerate(ensemble_state_dict):
            new_state_dict[key] = ensemble_state_dict[key].clone()
            new_state_dict[key][index] = single_state_dict[key].clone()

        self.load_state_dict(new_state_dict)


def test_ensemble_sparse_layer():
    batch_size, ensemble_size, input_dim, output_dim = 128, 7, 32, 64
    node_num = 10

    inputs = torch.randn(batch_size, node_num, input_dim)
    inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.repeat([1, ensemble_size, 1, 1])
    weights = torch.rand(ensemble_size, node_num, input_dim, output_dim)
    outputs = torch.einsum("abcd,bcde->abce", inputs, weights)

    ensemble_layer = EnsembleSparseLayer(input_dim, output_dim, ensemble_size=ensemble_size,
                                         node_num=node_num, bias=False)
    ensemble_layer.weight.data[:] = weights
    output_torch = ensemble_layer(inputs)

    # compare
    print(torch.allclose(output_torch, outputs))


def test_ensemble_sparse_mlp():
    batch_size = 256
    input_dim = 20
    output_dim = 10
    hidden_dims = [100, 100]
    ensemble_size = 7
    model = EnsembleSparseMLP(input_dim, 2, output_dim, hidden_dims, ensemble_size=ensemble_size)

    # print(model)

    inputs = torch.randn(batch_size, input_dim)
    inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.repeat([1, ensemble_size, 1])

    outputs = model(inputs)
    l1_loss = model.compute_l1_loss()
    l2_loss = model.compute_l2_loss([0.001, 0.001, 0.001])

    print(outputs.shape)
    print(l1_loss)
    print(l2_loss)


def test_ensemble_sparse_model():
    import gym
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 1
    hidden_dims = [200, 200]
    ensemble_size = 7

    model = EnsembleSparseModel(state_dim, action_dim, reward_dim, hidden_dims, ensemble_size)

    batch_size = 256
    sampled_states = torch.randn([batch_size, ensemble_size, state_dim])
    sampled_actions = torch.randn([batch_size, ensemble_size, action_dim])

    outputs = model(sampled_states, sampled_actions)
    print(outputs[0]["diff_states"].shape)


if __name__ == "__main__":
    setattr(torch, 'identity', lambda x: x)
    setattr(torch, 'swish', lambda x: x * torch.sigmoid(x))

    test_ensemble_sparse_model()