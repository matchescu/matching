from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import Module, CrossEntropyLoss, Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._module import MultiClassModule
from .._params import MultiClassTrainingParams
from ._config import CAPABILITY
from ._datasets import AsymmetricMultiClassDataset


class MultiClassTrainer(
    BaseTrainer[
        MultiClassModule, MultiClassTrainingParams, AsymmetricMultiClassDataset
    ],
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

        counts = torch.tensor(
            [
                hyperparams.neg_pos_ratio * (hyperparams.match_bridge_ratio + 2),
                hyperparams.match_bridge_ratio,
                1,
                1,
            ]
        )
        inv_counts = 1.0 / counts
        weights = inv_counts / inv_counts.sum()

        super().__init__(
            task_name,
            hyperparams,
            model_dir or Path(__file__).parent,
            loss_fn or CrossEntropyLoss(weight=weights),
            **kwargs,
        )

    def _setup_model(self, model: MultiClassModule) -> MultiClassModule:
        return model.with_frozen_bert_layers(self._params.frozen_layer_count)

    @classmethod
    def _get_decay_params(
        cls, model: Module, has_decaying_weights: bool = True
    ) -> list[Parameter]:
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
        return [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay) != has_decaying_weights
            and p.requires_grad
        ]

    @classmethod
    def _get_decay_model_params(
        cls, model: Module, learning_rate: float, weight_decay: float
    ) -> Iterable[dict]:
        decaying_params = cls._get_decay_params(model, True)
        non_decaying_params = cls._get_decay_params(model, False)
        yield {
            "params": decaying_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        yield {
            "params": non_decaying_params,
            "lr": learning_rate,
            "weight_decay": 0.0,
        }

    def _create_optimizer(self, model: MultiClassModule) -> Optimizer:
        base_lr = self._params.learning_rate
        decay_factor = self._params.decay_factor
        weight_decay = self._params.weight_decay

        num_layers = len(model.encoder_layers)
        param_groups = []

        # classification head
        param_groups.extend(
            self._get_decay_model_params(
                model.classifier, base_lr / decay_factor, weight_decay
            )
        )

        # encoder layers (for those that were frozen, decay to 0)
        for i, layer in enumerate(reversed(model.encoder_layers)):
            layer_lr = base_lr * (decay_factor**i)
            param_groups.extend(
                self._get_decay_model_params(layer, layer_lr, weight_decay)
            )

        # lowest learning rate for embeddings
        emb_lr = base_lr * (decay_factor**num_layers)
        param_groups.extend(
            self._get_decay_model_params(model.embeddings_layer, emb_lr, weight_decay)
        )

        return torch.optim.AdamW(param_groups)

    def _create_scheduler(
        self, dataset: AsymmetricMultiClassDataset, optimizer: Optimizer
    ):
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
