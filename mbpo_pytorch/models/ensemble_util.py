import torch
import torch.nn as nn
import math
import numpy as np

from collections import OrderedDict

from typing import List, Dict, Union, Optional


from abc import ABC


class EnsembleLayer(nn.Module):
    def __init__(self, input_features, output_features, ensemble_size=7, bias=True):
        super(EnsembleLayer, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(ensemble_size,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_features))
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

    def forward(self, input: torch.Tensor):
        # input-size: batch-size * ensemble-size * input-dim
        # weight: ensemble-size * input-dim * output-dim
        out = torch.einsum("abc,bcd->abd", input, self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):
        return 'EnsembleLayer(ensemble_size={}, input_features={}, output_features={}, bias={})'.format(
            self.ensemble_size, self.input_features, self.output_features,
            self.bias is not None)

    def __str__(self):
        return 'ensemble_size={}, input_features={}, output_features={}, bias={}'.format(
            self.ensemble_size, self.input_features, self.output_features,
            self.bias is not None)


class EnsembleMLP(nn.Module, ABC):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 activation='tanh',
                 last_activation='identity',
                 ensemble_size=7,
                 biases=None):
        super(EnsembleMLP, self).__init__()
        sizes_list = hidden_dims.copy()
        self.activation = getattr(torch, activation)
        # setattr in __init__.py of models
        self.last_activation = getattr(torch, last_activation)
        self.ensemble_size = ensemble_size

        sizes_list.insert(0, input_dim)
        biases = [True] * len(sizes_list) if biases is None else biases.copy()

        layers = []
        if 1 < len(sizes_list):
            for i in range(len(sizes_list) - 1):
                layers.append(EnsembleLayer(sizes_list[i], sizes_list[i + 1], ensemble_size, bias=biases[i]))
        self.last_layer = EnsembleLayer(sizes_list[-1], output_dim, ensemble_size)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        x = self.last_activation(x)
        return x

    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer) + "\n"
            string += str(self.activation) + "\n"
        string += str(self.last_layer) + "\n"
        string += str(self.last_activation)
        return string

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
        weight_norms = []
        for name, weight in self.named_parameters():
            if "weight" in name:
                weight_norms.append(weight.norm(2))
        weight_norms = torch.stack(weight_norms, dim=0)
        weight_decay = (torch.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay


class EnsembleModel(EnsembleMLP):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int],
                 output_state_dim=None, ensemble_size=7):
        input_dim = state_dim + action_dim
        output_state_dim = output_state_dim or state_dim
        output_dim = (output_state_dim + reward_dim) * 2

        self.output_state_dim = output_state_dim
        self.ensemble_size = ensemble_size
        super(EnsembleModel, self).__init__(input_dim, output_dim, hidden_dims, ensemble_size=ensemble_size)

    def forward(self, states, actions):
        # input size: batch-size * ensemble-size * dim
        inputs = torch.cat([states, actions], dim=-1)
        # print(inputs.shape)
        outputs = super(EnsembleModel, self).forward(inputs)
        # print(outputs.shape)
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


def test_ensemble_layer():
    batch_size, ensemble_size, input_dim, output_dim = 128, 7, 32, 64

    inputs = torch.randn(batch_size, input_dim)
    inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.repeat([1, ensemble_size, 1])
    weights = torch.rand(ensemble_size, input_dim, output_dim)
    outputs = torch.einsum("abc,bcd->abd", inputs, weights)

    ensemble_layer = EnsembleLayer(input_dim, output_dim, ensemble_size=ensemble_size, bias=False)
    ensemble_layer.weight.data[:] = weights
    output_torch = ensemble_layer(inputs)

    # compare
    print(torch.allclose(output_torch, outputs))

def test_ensemble_mlp():
    batch_size = 256
    input_dim = 20
    output_dim = 1
    hidden_dims = [100, 100]
    ensemble_size = 5
    model = EnsembleMLP(input_dim, output_dim, hidden_dims, ensemble_size=ensemble_size)

    # print(model)

    inputs = torch.randn(batch_size, input_dim)
    inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.repeat([1, ensemble_size, 1])

    outputs = model(inputs)

    print(outputs.shape)


def test_ensemble_model():
    import gym
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 1
    hidden_dims = [200, 200]
    ensemble_size = 7

    model = EnsembleModel(state_dim, action_dim, reward_dim, hidden_dims, ensemble_size)

    state_dict = model.get_state_dict(2)
    model.set_state_dict(2, state_dict)



if __name__ == '__main__':
    setattr(torch, 'identity', lambda x: x)
    setattr(torch, 'swish', lambda x: x * torch.sigmoid(x))

    # test_ensemble_model()

    test_ensemble_mlp()
