import torch
import torch.nn as nn
import math
import numpy as np

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

    print(output_torch.shape)


if __name__ == '__main__':
    test_ensemble_mlp()
