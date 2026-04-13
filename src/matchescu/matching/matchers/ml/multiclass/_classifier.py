from typing import Self

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_p: float = 0.1,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, output_size, dtype=dtype),
        )

    def forward(self, x):
        return self._model(x)

    def to(self, device: str | torch.device) -> Self:
        super().to(device)
        self._model.to(device)
        return self

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self._model.train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        self._model.eval()
        return self
