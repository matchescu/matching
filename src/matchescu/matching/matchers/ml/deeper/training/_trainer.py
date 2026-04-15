from os import PathLike
from typing import Any

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseTrainer, TDataset
from .._module import DeepERModule
from .._params import DeepERParams
from ._config import CAPABILITY
from ._dataset import DeepERDataset


class DeepERTrainer(
    BaseTrainer[DeepERModule, DeepERParams, DeepERDataset],
    capability=CAPABILITY,
):
    hyperparams_schema = DeepERParams

    def __init__(
        self,
        task_name: str,
        hyperparams: DeepERParams,
        model_dir: str | PathLike | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(task_name, hyperparams, model_dir, **kwargs)

    @classmethod
    def _create_loss(cls, _: DataLoader[TDataset]) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    @classmethod
    def _forward_pass(
        cls,
        model: DeepERModule,
        batch: tuple[
            list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], torch.Tensor
        ],
        device: torch.device,
    ) -> tuple:
        lhs_attrs, rhs_attrs, labels = batch
        lhs_attrs = [{k: v.to(device) for k, v in attr.items()} for attr in lhs_attrs]
        rhs_attrs = [{k: v.to(device) for k, v in attr.items()} for attr in rhs_attrs]
        labels = labels.to(device)

        return model(left_attrs=lhs_attrs, right_attrs=rhs_attrs), labels
