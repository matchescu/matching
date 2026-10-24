from typing import Self

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_p: float = 0.1,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self._input_size = input_size
        self._head_p = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 2, dtype=dtype),
        )
        self._head_q = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 2, dtype=dtype),
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    def forward(self, x):
        return torch.cat([self._head_p(x), self._head_q(x)], dim=-1)

    def to(self, device: str | torch.device) -> Self:
        super().to(device)
        self._head_p.to(device)
        self._head_q.to(device)
        return self

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self._head_p.train(mode)
        self._head_q.train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        self._head_p.eval()
        self._head_q.eval()
        return self
