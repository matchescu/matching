from typing import Self

import torch
import torch.nn as nn


class ResidualHead(nn.Sequential):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout_p: float = 0.1,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.dropout(self.act(self.fc1(x)))
        x = x + residual  # skip connection
        return self.fc2(x)

    def to(self, device: str | torch.device) -> Self:
        super().to(device)
        self.fc1.to(device)
        self.act.to(device)
        self.dropout.to(device)
        self.fc2.to(device)
        return self

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.fc1.train(mode)
        self.act.train(mode)
        self.dropout.train(mode)
        self.fc2.train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        self.fc1.eval()
        self.act.eval()
        self.dropout.eval()
        self.fc2.eval()
        return self
