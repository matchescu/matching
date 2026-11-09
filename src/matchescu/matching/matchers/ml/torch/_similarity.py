from abc import ABCMeta, abstractclassmethod, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Generic, Self

import torch

from matchescu.matching.matchers.ml.core import (
    AdditionalModelInfo,
    ModelLoadingError,
    TModel,
    TParams,
)
from matchescu.matching.matchers.ml.transformers import (
    suppress_transformer_modeling_utils_warnings,
)


class TorchMatcher(Generic[TModel, TParams], metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._model: TModel | None = None

    @abstractmethod
    def _validate(self, path: Path, info: dict) -> AdditionalModelInfo[TParams]:
        raise NotImplementedError

    @abstractclassmethod
    def _new_model(cls, info: AdditionalModelInfo[TParams]) -> TModel:
        raise NotImplementedError

    def load_from_file(self, path: str | PathLike) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        model_dict = torch.load(path, weights_only=False)
        if (additional_info := model_dict.get("additional_info")) is None:
            raise ModelLoadingError(
                path, "expected hyperparameters to be bundled with model"
            )
        additional_info = self._validate(path, additional_info)

        try:
            with suppress_transformer_modeling_utils_warnings():
                self._model = self._new_model(additional_info)
                self._model.load_state_dict(model_dict["model"], strict=True)
                self._model.training = False
                self._model.train(False)
                self._model.eval()
        except Exception as e:
            raise ModelLoadingError(path, str(e)) from e
        return self
