from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.matching.matchers.ml.ditto._module import DittoModel
from matchescu.matching.matchers.ml.ditto._encoder import to_ditto_text
from matchescu.matching.matchers.ml.ditto._params import DittoModelTrainingParams
from matchescu.matching.matchers.ml.ditto.training._evaluator import TrainingEvaluator
from matchescu.matching.matchers.ml.transformers import (
    suppress_transformer_modeling_utils_warnings,
)
from matchescu.matching.similarity import Similarity
from matchescu.similarity import MatchResult
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
        non_match = MatchResult(0, [1, 0])
        super().__init__(non_match, non_match)
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

    def load_from_file(self, path: str | PathLike) -> "DittoSimilarity":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)
        model_dict = torch.load(path)

        if (additional_info := model_dict.get("additional_info")) is None:
            raise ValueError("expected hyperparameters to be bundled with model")
        additional_info = AdditionalModelInfo[DittoModelTrainingParams].model_validate(
            additional_info
        )

        if TrainingEvaluator.THRESHOLD_KEY not in additional_info.best_config:
            raise ValueError("expected best threshold to be bundled with model")
        self.__threshold = float(
            additional_info.best_config[TrainingEvaluator.THRESHOLD_KEY]
        )

        with suppress_transformer_modeling_utils_warnings():
            self.__model = DittoModel(additional_info.hyperparameters)
            self.__model.load_state_dict(model_dict["model"], strict=True)
            self.__model.training = False
            self.__model.train(False)
            self.__model.eval()
        return self

    def _compute_similarity(
        self, a: EntityReference, b: EntityReference
    ) -> MatchResult:
        with torch.no_grad():
            text_a = to_ditto_text(a, self.__left_cols)
            text_b = to_ditto_text(b, self.__right_cols)
            encoded_text = torch.LongTensor(
                self.__tokenizer.encode(
                    text=text_a,
                    text_pair=text_b,
                    max_length=self.__max_len,
                    truncation=True,
                )
            ).unsqueeze(0)
            weight = torch.sigmoid(self.__model(encoded_text)).item()
        prediction = 1 if weight > self.__threshold else 0
        return MatchResult(prediction, [1 - weight, weight])
