from os import PathLike
from pathlib import Path
from typing import Iterable

import torch
from torch.distributions.utils import logits_to_probs
from transformers import PreTrainedTokenizerBase

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference

from ._encoder import to_deeper_repr, ensure_attr_map
from ._module import DeepERModule
from ._params import DeepERParams


class DeepERSimilarity:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        attr_map: dict | None = None,
        max_len: int = 30,
        excluded_attrs: Iterable[str | int] | None = None,
    ) -> None:
        self._model = None
        self._tokenizer = tokenizer
        self._attrs = attr_map
        self._max_len = max_len
        self._excluded = set(excluded_attrs or ())

    def load_from_file(self, path: str | PathLike) -> "DeepERSimilarity":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        model_dict = torch.load(path)
        if (additional_info := model_dict.get("additional_info")) is None:
            raise ValueError("expected hyperparameters to be bundled with model")

        additional_info = AdditionalModelInfo[DeepERParams].model_validate(
            additional_info
        )
        self._model = DeepERModule(additional_info.hyperparameters)
        self._model.load_state_dict(model_dict["model"], strict=True)
        return self

    def __call__(self, a: EntityReference, b: EntityReference) -> MatchResult:
        if self._model is None:
            raise RuntimeError("load model before calling")
        if not all(isinstance(r, EntityReference) for r in (a, b)):
            raise TypeError("both parameters must be entity references")

        attr_map = ensure_attr_map(a, b, self._attrs, self._excluded)

        with torch.no_grad():
            lhs, rhs = to_deeper_repr(a, b, self._tokenizer, attr_map, self._max_len)
            result = self._model(lhs, rhs).squeeze(0)
            prediction = torch.argmax(result).item()
        label_weights = logits_to_probs(result).tolist()
        return MatchResult(a.id, b.id, prediction, label_weights)
