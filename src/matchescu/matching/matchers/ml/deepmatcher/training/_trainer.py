from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from matchescu.matching.matchers.ml.deepmatcher import DeepMatcherModule
from matchescu.matching.matchers.ml.deepmatcher.training._dataset import (
    DeepMatcherDataset,
)
from matchescu.matching.matchers.ml.training import BaseTrainer, ModelTrainingParams
from matchescu.matching.matchers.ml.training._typevars import TModel


class DeepMatcherTrainer(
    BaseTrainer[DeepMatcherModule, ModelTrainingParams, DeepMatcherDataset],
    capability="deepmatcher",
):
    def __init__(
        self,
        task_name: str,
        hyperparams: ModelTrainingParams,
        model_dir: str | PathLike | None = None,
        loss_fn: _Loss | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_name,
            model_dir,
            hyperparams,
            loss_fn or nn.CrossEntropyLoss(),
            **kwargs,
        )

    @classmethod
    def _forward_pass(cls, model: TModel, batch: Any, device: torch.device) -> tuple:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)

        return model(**inputs), labels
