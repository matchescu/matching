from typing import Callable

from torch import nn, Tensor

from matchescu.matching.ml.modules._highway import HighwayNetwork


class HighwayMatchClassifier(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        layers: int = 2,
        highway_hidden_activation: Callable[[Tensor], Tensor] | None = None,
        highway_output_activation: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.add_module(
            "highway-net",
            HighwayNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=2,
                layers=layers,
                activation=highway_hidden_activation,
                initial_bias=-2,
                output_activation=highway_output_activation,
            )
        )
        self.add_module("softmax", nn.LogSoftmax(dim=-1))
