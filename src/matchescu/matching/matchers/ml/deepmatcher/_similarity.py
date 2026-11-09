from collections.abc import Iterable
from pathlib import Path

import torch
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference
from torch.distributions.utils import logits_to_probs
from transformers import PreTrainedTokenizerBase

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.matching.matchers.ml.deepmatcher._encoder import (
    ensure_attr_map,
    to_deepmatcher_repr,
)

from ..torch import TorchMatcher
from ._module import DeepMatcherModule
from ._params import DeepMatcherModelTrainingParams


class DeepMatcherSimilarity(
    TorchMatcher[DeepMatcherModule, DeepMatcherModelTrainingParams]
):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        attr_map: dict | None = None,
        max_len: int = 30,
        excluded_attrs: Iterable[str | int] | None = None,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._attrs = attr_map
        self._max_len = max_len
        self._excluded = set(excluded_attrs or ())

    def _validate(
        self, path: Path, info: dict
    ) -> AdditionalModelInfo[DeepMatcherModelTrainingParams]:
        return AdditionalModelInfo[DeepMatcherModelTrainingParams].model_validate(info)

    @classmethod
    def _new_model(
        cls, info: AdditionalModelInfo[DeepMatcherModelTrainingParams]
    ) -> DeepMatcherModule:
        return DeepMatcherModule(info.hyperparameters)

    def __call__(self, a: EntityReference, b: EntityReference) -> MatchResult:
        if self._model is None:
            raise RuntimeError("load model before calling")
        if not all(isinstance(r, EntityReference) for r in (a, b)):
            raise TypeError("both parameters must be entity references")

        attr_map = ensure_attr_map(a, b, self._attrs, self._excluded)

        with torch.no_grad():
            tokens = {
                k: v.unsqueeze(0)
                for k, v in to_deepmatcher_repr(
                    a, b, self._tokenizer, attr_map, self._max_len
                ).items()
            }
            result = self._model(**tokens).squeeze(0)
            prediction = torch.argmax(result).item()
        label_weights = logits_to_probs(result).tolist()
        return MatchResult(a.id, b.id, prediction, label_weights)
