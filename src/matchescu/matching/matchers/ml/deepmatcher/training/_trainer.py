from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._module import DeepMatcherModule
from .._params import DeepMatcherModelTrainingParams
from ._dataset import DeepMatcherDataset


class DeepMatcherTrainer(
    BaseTrainer[DeepMatcherModule, DeepMatcherModelTrainingParams, DeepMatcherDataset],
    capability="deepmatcher",
):
    hyperparams_schema = DeepMatcherModelTrainingParams

    def __init__(
        self,
        task_name: str,
        hyperparams: DeepMatcherModelTrainingParams,
        model_dir: str | PathLike | None = None,
        loss_fn: _Loss | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_name,
            hyperparams,
            model_dir,
            loss_fn or nn.CrossEntropyLoss(),
            **kwargs,
        )

    @classmethod
    def _forward_pass(
        cls, model: DeepMatcherModule, batch: Any, device: torch.device
    ) -> tuple:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)

        return model(**inputs), labels
