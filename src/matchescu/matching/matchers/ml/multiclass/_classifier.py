from typing import Self

import torch
from torch import nn

from ._loss import order_energy


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

    @property
    def input_size(self) -> int:
        return self._model[0].in_features

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


class AsymmetricHead(nn.Module):
    """Classify ordered pairs with independently projected interactions.

    .. note::
        Build pair features with :meth:`features` before calling :meth:`forward`.
        Both projections belong to this module so classifier optimizer groups
        and device movement include them alongside the existing classification MLP.

    :param hidden: Encoder width, also used for the MLP's hidden layer.
    :param rank: Interaction width. Defaults to 256.
    :param use_bilinear: Include independently projected products. Defaults to True.
    :param use_order: Include reverse-minus-forward order energy. Defaults to True.
    :param dropout_p: Dropout probability in the MLP. Defaults to 0.1.
    :param dtype: Initialize all weights in the encoder's floating-point dtype.
    """

    def __init__(
        self,
        hidden: int,
        rank: int = 256,
        use_bilinear: bool = True,
        use_order: bool = True,
        dropout_p: float = 0.1,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.U = (
            nn.Linear(hidden, rank, bias=False, dtype=dtype) if use_bilinear else None
        )
        self.V = (
            nn.Linear(hidden, rank, bias=False, dtype=dtype) if use_bilinear else None
        )
        self._use_order = use_order
        input_size = 3 * hidden + (rank if use_bilinear else 0) + int(use_order)
        self._classifier = ClassificationHead(input_size, hidden, 3, dropout_p, dtype)

    @property
    def input_size(self) -> int:
        return self._classifier.input_size

    def features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Build the ordered pair representation without classifying it.

        .. note::
            Concatenate ``[a, b, abs(a-b)]``, then ``U(a)*V(b)`` if enabled,
            then ``order_energy(b,a)-order_energy(a,b)`` as one scalar if enabled.
            Only the energy delta is necessarily antisymmetric under pair reversal.
        """
        parts = [a, b, torch.abs(a - b)]
        if self.U is not None:
            parts.append(self.U(a) * self.V(b))
        if self._use_order:
            parts.append((order_energy(b, a) - order_energy(a, b)).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify precomputed features without rebuilding the pair representation."""
        return self._classifier(x)
