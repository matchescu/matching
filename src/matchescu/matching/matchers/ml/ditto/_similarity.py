from collections.abc import Iterable
from os import PathLike, curdir
from pathlib import Path

import torch
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.core import AdditionalModelInfo, ModelLoadingError
from matchescu.matching.matchers.ml.ditto._encoder import to_ditto_text
from matchescu.matching.matchers.ml.ditto._module import DittoModel
from matchescu.matching.matchers.ml.ditto._params import DittoModelTrainingParams
from matchescu.matching.matchers.ml.ditto.training._evaluator import TrainingEvaluator
from matchescu.matching.matchers.ml.torch import TorchMatcher


class DittoSimilarity(TorchMatcher[DittoModel, DittoModelTrainingParams]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        threshold: float = 0.5,
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
        self.__threshold = threshold
        self.__left_cols = left_cols
        self.__right_cols = right_cols

    @property
    def match_threshold(self) -> float:
        return self.__threshold

    def _validate(
        self, path: Path, info: dict
    ) -> AdditionalModelInfo[DittoModelTrainingParams]:
        ret = AdditionalModelInfo[DittoModelTrainingParams].model_validate(info)
        if TrainingEvaluator.THRESHOLD_KEY not in ret.best_config:
            raise ModelLoadingError(
                path, "expected best threshold to be bundled with model"
            )
        self.__threshold = float(ret.best_config[TrainingEvaluator.THRESHOLD_KEY])
        return ret

    @classmethod
    def _new_model(
        cls, info: AdditionalModelInfo[DittoModelTrainingParams]
    ) -> DittoModel:
        return DittoModel(info.hyperparameters)

    def __call__(self, a: EntityReference, b: EntityReference) -> MatchResult:
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
            weight = torch.sigmoid(self._model(encoded_text)).item()
        prediction = 1 if weight > self.__threshold else 0
        return MatchResult(a.id, b.id, prediction, [1 - weight, weight])
