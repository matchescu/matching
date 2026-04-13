from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from torch.distributions.utils import logits_to_probs
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.matching.matchers.ml.transformers import (
    suppress_transformer_modeling_utils_warnings,
)
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference

from ._encoder import to_ditto_text
from ._module import MultiClassModule
from ._params import MultiClassTrainingParams


class MultiClassSimilarity:
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
        self.__left_cols = left_cols
        self.__right_cols = right_cols

    def load_from_file(self, path: str | PathLike) -> "MultiClassSimilarity":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)
        model_dict = torch.load(path)

        if (additional_info := model_dict.get("additional_info")) is None:
            raise ValueError("expected hyperparameters to be bundled with model")
        additional_info = AdditionalModelInfo[MultiClassTrainingParams].model_validate(
            additional_info
        )

        with suppress_transformer_modeling_utils_warnings():
            self.__model = MultiClassModule(additional_info.hyperparameters)
            self.__model.load_state_dict(model_dict["model"], strict=True)
            self.__model.training = False
            self.__model.train(False)
            self.__model.eval()
        return self

    def __call__(self, a: EntityReference, b: EntityReference) -> MatchResult:
        with torch.no_grad():
            text_a = to_ditto_text(a, self.__left_cols)
            text_b = to_ditto_text(b, self.__right_cols)
            encoding = self.__tokenizer(
                text=text_a,
                text_pair=text_b,
                max_length=self.__max_len,
                truncation=True,
                return_tensors="pt",
            )
            cls_logits = self.__model(**encoding).squeeze(0)
            prediction: int = torch.argmax(cls_logits, dim=-1).int().item()
            cls_weights = logits_to_probs(cls_logits).tolist()
        return MatchResult(a.id, b.id, prediction, cls_weights)
