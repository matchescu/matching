from typing import Any, Callable

import torch

from matchescu.matching.similarity import Similarity


class TensorSimilarity(Similarity):
    def __init__(
        self,
        similarity_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        self._compute_distance = similarity_func or torch.nn.PairwiseDistance()

    def _compute_similarity(self, a: Any, b: Any) -> float:
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        assert len(a.shape) == 1 or (len(a.shape) == 2 and any(a.shape) == 1)
        t = self._compute_distance(a, b)
        return t.detach().item()


class TensorDiff:
    def __init__(
        self, diff_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
    ) -> None:
        self._diff = diff_func or torch.sub

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        assert len(a.shape) == 1 or (len(a.shape) == 2 and any(a.shape) == 1)
        return self._diff(a, b)
