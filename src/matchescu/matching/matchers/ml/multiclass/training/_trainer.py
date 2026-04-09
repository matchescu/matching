from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.nn import BCEWithLogitsLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._module import MultiClassModule
from .._params import MultiClassTrainingParams
from ._config import CAPABILITY
from ._datasets import DittoDataset


class MultiClassTrainer(
    BaseTrainer[MultiClassModule, MultiClassTrainingParams, DittoDataset],
    capability=CAPABILITY,
):
    hyperparams_schema = MultiClassTrainingParams

    def __init__(
        self,
        task_name: str,
        hyperparams: MultiClassTrainingParams,
        model_dir: str | PathLike | None = None,
        loss_fn: _Loss | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_name,
            hyperparams,
            model_dir or Path(__file__).parent,
            loss_fn or BCEWithLogitsLoss(),
            **kwargs,
        )

    def _setup_model(self, model: MultiClassModule) -> MultiClassModule:
        return model.with_frozen_bert_layers(self._params.frozen_layer_count)

    def _create_optimizer(self, model: MultiClassModule):
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

        param_groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self._params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=self._params.learning_rate)

    def _create_scheduler(self, dataset: DittoDataset, optimizer: Optimizer):
        total_batches = len(dataset) // self._params.batch_size
        num_steps = total_batches * self._params.epochs

        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_steps
        )

    @classmethod
    def _forward_pass(cls, model: Module, batch: tuple, device: torch.device) -> tuple:
        device_batch = tuple(item.to(device) for item in batch)
        if len(device_batch) == 2:
            x, y = device_batch
            prediction = model(x.to(device))
        else:
            x1, x2, y = device_batch
            prediction = model(x1.to(device), x2.to(device))
        return prediction, y
