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
        if last_activation == 'identity':
            self.last_activation = lambda x: x
        else:
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


class EnsembleModel:
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int],
                 output_state_dim=None, ensemble_size=7):
        input_dim = state_dim + action_dim
        output_state_dim = output_state_dim or state_dim
        output_dim = (output_state_dim + reward_dim) * 2

        self.output_state_dim = output_state_dim
        self.ensemble_size = ensemble_size
        self.ensemble_mlp = EnsembleMLP(input_dim, output_dim, hidden_dims, ensemble_size=ensemble_size)

    def __call__(self, *args, **kwargs):
        return self.forward_mlp(*args, **kwargs)

    def forward_mlp(self, states, actions):
        inputs = torch.cat([states, actions], dim=1)
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.repeat(1, self.ensemble_size, 1)
        # print(inputs.shape)
        outputs = self.ensemble_mlp(inputs)
        # print(outputs.shape)
        outputs = [{'diff_states': outputs[:, i, :self.output_state_dim * 2],
              'rewards': outputs[:, i, self.output_state_dim * 2:]}
             for i in range(self.ensemble_size)]
        return outputs

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
        weight_norms = []
        for name, weight in self.ensemble_mlp.named_parameters():
            if "weight" in name:
                weight_norms.append(weight.norm(2))
        weight_norms = torch.stack(weight_norms, dim=0)
        weight_decay = (torch.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay

    def get_state_dict(self, index):
        assert index < self.ensemble_size

        ensemble_state_dict = self.ensemble_mlp.state_dict().copy()
        single_state_dict = OrderedDict()
        for i, key in enumerate(ensemble_state_dict):
            single_state_dict[key] = ensemble_state_dict[key][index]

        return single_state_dict

    def load_state_dict(self, index, single_state_dict):
        assert index < self.ensemble_size

        new_state_dict = OrderedDict()
        ensemble_state_dict = self.ensemble_mlp.state_dict().copy()
        for i, key in enumerate(ensemble_state_dict):
            new_state_dict[key] = ensemble_state_dict[key]
            new_state_dict[key][index] = single_state_dict[key][index]

        self.ensemble_mlp.load_state_dict(new_state_dict)


def test_ensemble_layer():
    batch_size, ensemble_size, input_dim, output_dim = 128, 7, 32, 64

    # numpy
    input_numpy = np.random.randn(batch_size, input_dim)
    input_numpy = np.expand_dims(input_numpy, 1)
    input_numpy = np.tile(input_numpy, [1, ensemble_size, 1])
    weight = np.random.randn(ensemble_size, input_dim, output_dim)
    output_numpy = np.einsum("abc,bcd->abd", input_numpy, weight)

    # torch
    input_torch = torch.from_numpy(input_numpy).type(dtype='torch.FloatTensor')
    ensemble_layer = EnsembleLayer(ensemble_size, input_dim, output_dim, bias=False)
    ensemble_layer.weight.data[:] = torch.from_numpy(weight).type(dtype='torch.FloatTensor')
    output_torch = ensemble_layer(input_torch)

    # compare
    print(torch.allclose(output_torch, torch.from_numpy(output_numpy)))

def test_ensemble_mlp():
    batch_size = 256
    input_dim = 20
    output_dim = 1
    hidden_dims = [100, 100]
    ensemble_size = 5
    model = EnsembleMLP(input_dim, output_dim, hidden_dims, ensemble_size=ensemble_size)

    # print(model)

    input_numpy = np.random.randn(batch_size, input_dim)
    input_numpy = np.expand_dims(input_numpy, 1)
    input_numpy = np.tile(input_numpy, [1, ensemble_size, 1])

    # torch.set_default_dtype(torch.double)
    input_torch = torch.from_numpy(input_numpy).type(dtype='torch.FloatTensor')
    output_torch = model(input_torch)

    # print(output_torch.shape)


def test_ensemble_model():
    import gym
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 1
    hidden_dims = [200, 200]
    ensemble_size = 7

    model = EnsembleModel(state_dim, action_dim, reward_dim, hidden_dims, ensemble_size)

    model.get_state_dict(2)



if __name__ == '__main__':
    test_ensemble_model()
