from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.ditto._ditto_module import DittoModel
from matchescu.matching.matchers.ml.ditto._ref_adapter import to_text
from matchescu.matching.similarity import Similarity
from matchescu.typing import EntityReference


class DittoSimilarity(Similarity):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 256,
        model_dir: str | PathLike | None = None,
        left_cols: Iterable[str] | None = None,
        right_cols: Iterable[str] | None = None,
    ) -> None:
        model_dir = model_dir or curdir
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__model_dir = Path(model_dir)
        self.__model = None
        self.__threshold = 0.5
        self.__left_cols = left_cols
        self.__right_cols = right_cols

    @property
    def non_match_threshold(self) -> float:
        return 1 - self.__threshold

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

    def _compute_similarity(self, a: EntityReference, b: EntityReference) -> float:
        with torch.no_grad():
            encoded_text = torch.LongTensor(
                self.__tokenizer.encode(
                    text=to_text(a, self.__left_cols),
                    text_pair=to_text(b, self.__right_cols),
                    max_length=self.__max_len,
                    truncation=True,
                )
            ).unsqueeze(0)
            return torch.sigmoid(self.__model(encoded_text)).item()
