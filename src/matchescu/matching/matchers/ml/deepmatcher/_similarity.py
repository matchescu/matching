from os import PathLike
from pathlib import Path
from typing import Iterable

import torch
from torch.distributions.utils import logits_to_probs
from transformers import PreTrainedTokenizerBase

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.matching.matchers.ml.deepmatcher._encoder import (
    to_deepmatcher_repr,
    ensure_attr_map,
)
from matchescu.matching.similarity import Similarity
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference

from ._module import DeepMatcherModule
from ._params import DeepMatcherModelTrainingParams


class DeepMatcherSimilarity(Similarity[MatchResult]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        attr_map: dict | None = None,
        max_len: int = 30,
        excluded_attrs: Iterable[str | int] | None = None,
    ) -> None:
        non_match = MatchResult(0, [1, 0])
        super().__init__(non_match, non_match)
        self._model = None
        self._tokenizer = tokenizer
        self._attrs = attr_map
        self._max_len = max_len
        self._excluded = set(excluded_attrs or ())

    def load_from_file(self, path: str | PathLike) -> "DeepMatcherSimilarity":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        model_dict = torch.load(path)
        if (additional_info := model_dict.get("additional_info")) is None:
            raise ValueError("expected hyperparameters to be bundled with model")

        additional_info = AdditionalModelInfo[
            DeepMatcherModelTrainingParams
        ].model_validate(additional_info)
        self._model = DeepMatcherModule(additional_info.hyperparameters)
        return self

    def _compute_similarity(
        self, a: EntityReference, b: EntityReference
    ) -> MatchResult:
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
        return MatchResult(prediction, label_weights)
