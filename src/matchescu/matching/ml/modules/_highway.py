from typing import Callable

from torch import nn, Tensor
from torch.nn import functional as f


class HighwayLayer(nn.Module):
    def __init__(
        self,
        unit_count: int,
        activation: Callable[[Tensor], Tensor] | None = None,
        initial_bias: int = -2,
    ):
        super(HighwayLayer, self).__init__()
        self._activation = activation or f.relu
        self._basic_processor = nn.Linear(unit_count, unit_count)
        nn.init.xavier_uniform(self._basic_processor.weight)
        self._transform_gate = nn.Linear(unit_count, unit_count)
        self._transform_gate.bias.data.fill_(initial_bias)

    def forward(self, x):
        h = self._activation(self._basic_processor(x))
        t = f.sigmoid(self._transform_gate(x))

        return h*t + (1.0 - t)*x


class HighwayNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layers: int = 2,
        activation: Callable[[Tensor], Tensor] | None = None,
        initial_bias: int = -2,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        self._output_activation = output_activation or f.softmax

        self._scale_in = nn.Linear(input_size, hidden_size)
        self._layers = nn.ModuleList([
            HighwayLayer(hidden_size, activation, initial_bias) for _ in range(layers)
        ])
        self._scale_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = f.relu(self._scale_in(x))
        for highway_layer in self._layers:
            hidden = highway_layer(hidden)
        return self._output_activation(self._scale_out(hidden))