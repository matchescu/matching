from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers._result import MatchResult
from matchescu.matching.matchers.ml.ditto._module import DittoModel
from matchescu.matching.matchers.ml.ditto._encoder import to_ditto_text
from matchescu.matching.similarity import Similarity
from matchescu.typing import EntityReference


class DittoSimilarity(Similarity[MatchResult]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        threshold: float = 0.5,
        max_len: int = 256,
        model_dir: str | PathLike | None = None,
        left_cols: Iterable[str] | None = None,
        right_cols: Iterable[str] | None = None,
    ) -> None:
        super().__init__(0, 0)
        model_dir = model_dir or curdir
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__model_dir = Path(model_dir)
        self.__model = None
        self.__threshold = threshold
        self.__left_cols = left_cols
        self.__right_cols = right_cols

    @property
    def match_threshold(self) -> float:
        return self.__threshold

    def load_pretrained(self, model_name: str) -> None:
        model_path = self.__model_dir / model_name / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model_dict = torch.load(model_path)
        self.__threshold = float(model_dict["threshold"])
        self.__model = DittoModel(model_name)
        self.__model.load_state_dict(model_dict["model"])

    def _compute_similarity(
        self, a: EntityReference, b: EntityReference
    ) -> MatchResult:
        with torch.no_grad():
            encoded_text = torch.LongTensor(
                self.__tokenizer.encode(
                    text=to_ditto_text(a, self.__left_cols),
                    text_pair=to_ditto_text(b, self.__right_cols),
                    max_length=self.__max_len,
                    truncation=True,
                )
            ).unsqueeze(0)
            weight = torch.sigmoid(self.__model(encoded_text)).item()

        return MatchResult(1 if weight > self.__threshold else 0, weight)
