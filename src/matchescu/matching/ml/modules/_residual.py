import math

import torch
import torch.nn as nn


class ResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        residual_scale=True,
    ):
        """
        Initializes a residual network with multiple layers.

        This `article <https://arxiv.org/pdf/1512.03385>`_ provides more details
        on residual networks.

        Args:
            input_size (int): The size of the input vector.
            hidden_size (int): The size of the transformed vector.
            num_layers (int): Number of residual layers.
            residual_scale (bool): Whether to scale the residual output by sqrt(0.5).
        """
        super().__init__()

        self.__n_layers = num_layers
        self.__should_use_residual_scaling = residual_scale
        self.__layers = nn.ModuleList()
        self.__in_transform = nn.Linear(input_size, hidden_size)

        for _ in range(num_layers):
            self.__layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )

    def forward(self, x):
        """
        Forward pass through the residual network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, transformed_size).
        """
        residual = self.__in_transform(x)
        for layer in self.__layers:
            transformed = layer(residual)
            residual = transformed + residual  # Residual connection
            if self.__should_use_residual_scaling:
                residual *= math.sqrt(0.5)
        return residual
