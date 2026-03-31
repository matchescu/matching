from typing import Iterable

import torch
from transformers import PreTrainedTokenizerBase

from matchescu.matching.matchers.ml.core import MatchResult
from matchescu.matching.matchers.ml.deepmatcher._encoder import (
    to_deepmatcher_repr,
    ensure_attr_map,
)
from matchescu.matching.matchers.ml.deepmatcher._module import DeepMatcherModule
from matchescu.matching.similarity import Similarity
from matchescu.typing import EntityReference


class DeepMatcherSimilarity(Similarity[MatchResult]):
    def __init__(
        self,
        model: DeepMatcherModule,
        tokenizer: PreTrainedTokenizerBase,
        attr_map: dict | None = None,
        max_len: int = 30,
        excluded_attrs: Iterable[str | int] | None = None,
    ) -> None:
        super().__init__(0, 0)
        self._model = model
        self._tokenizer = tokenizer
        self._attrs = attr_map
        self._max_len = max_len
        self._excluded = set(excluded_attrs or ())

    def _compute_similarity(
        self, a: EntityReference, b: EntityReference
    ) -> MatchResult:
        if not all(isinstance(r, EntityReference) for r in (a, b)):
            raise TypeError("both parameters must be entity references")

        attr_map = ensure_attr_map(a, b, self._attrs, self._excluded)

        with torch.no_grad():
            tokens = to_deepmatcher_repr(a, b, self._tokenizer, attr_map, self._max_len)
            result = self._model(**tokens)
            prediction = torch.argmax(result)

        return MatchResult(1 if prediction > 0 else 0, result[prediction])
