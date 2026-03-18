"""Highway network implementation for deep feature transformation"""

import torch
import torch.nn as nn


class HighwayLayer(nn.Module):
    """Single highway network layer with gating mechanism"""

    def __init__(self, input_dim: int, dropout: float = 0.2):
        """
        Args:
            input_dim: Dimension of input features
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.input_dim = input_dim

        # Transform gate parameters
        self.W_T = nn.Linear(input_dim, input_dim)
        # Initial bias favors carry
        self.b_T = nn.Parameter(torch.ones(input_dim) * (-1.0))

        # Transform function parameters
        self.W_H = nn.Linear(input_dim, input_dim)
        self.b_H = nn.Parameter(torch.zeros(input_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Transformed tensor of same shape
        """
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.view(-1, self.input_dim)

        # Transform gate: determines how much of transformed input to use
        T = torch.sigmoid(self.W_T(x_reshaped) + self.b_T)

        # Transform function: non-linear transformation
        H = self.relu(self.W_H(x_reshaped) + self.b_H)
        H = self.dropout(H)

        # Carry connection: how much of original input to preserve
        output = H * T + x_reshaped * (1.0 - T)

        return output.view(batch_size, seq_len, self.input_dim)

    def train(self, mode: bool = True) -> "HighwayLayer":
        self.W_T.train(mode)
        self.W_H.train(mode)
        self.dropout.train(mode)
        self.relu.train(mode)
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self.W_T.eval()
        self.W_H.eval()
        self.dropout.eval()
        self.relu.eval()

    def to(self, device: str | torch.device) -> None:
        super().to(device)
        self.W_T.to(device)
        self.b_T.to(device)
        self.W_H.to(device)
        self.b_H.to(device)
        self.dropout.to(device)
        self.relu.to(device)


class HighwayNet(nn.Module):
    """Stacked highway network with configurable depth"""

    def __init__(self, input_dim: int, num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_dim: Dimension of input features
            num_layers: Number of highway layers to stack
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [HighwayLayer(input_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multiple highway layers"""
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, mode: bool = True) -> "HighwayNet":
        self.layers.train(mode)
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self.layers.eval()

    def to(self, device: str | torch.device) -> None:
        super().to(device)
        self.layers.to(device)
