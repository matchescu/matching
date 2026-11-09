from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.core import AdditionalModelInfo

from ..torch import TorchMatcher
from ._decoder import decode_logits, per_class_probabilities
from ._encoder import to_ditto_text
from ._module import MultiClassModule
from ._params import MultiClassTrainingParams


class MultiClassSimilarity(TorchMatcher[MultiClassModule, MultiClassTrainingParams]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 256,
        model_dir: str | PathLike | None = None,
        left_cols: Iterable[str] | None = None,
        right_cols: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        model_dir = model_dir or curdir
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__model_dir = Path(model_dir)
        self.__left_cols = left_cols
        self.__right_cols = right_cols

    def _validate(
        self, path: Path, info: dict
    ) -> AdditionalModelInfo[MultiClassTrainingParams]:
        return AdditionalModelInfo[MultiClassTrainingParams].model_validate(info)

    @classmethod
    def _new_model(
        cls, info: AdditionalModelInfo[MultiClassTrainingParams]
    ) -> MultiClassModule:
        return MultiClassModule(info.hyperparameters)

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
            col_token_id = self.__tokenizer("COL", add_special_tokens=False)[
                "input_ids"
            ][0]
            input_ids = encoding["input_ids"]
            positions = (input_ids == col_token_id).nonzero(as_tuple=False)
            col_positions = (
                positions.squeeze(-1).unsqueeze(0)
                if positions.numel() > 0
                else torch.tensor([[-1]], dtype=torch.long)
            )
            encoding["col_positions"] = col_positions
            cls_logits = self._model(**encoding)
            prediction: int = decode_logits(cls_logits).item()
            cls_weights = per_class_probabilities(cls_logits).squeeze(0).tolist()
        return MatchResult(a.id, b.id, prediction, cls_weights)
