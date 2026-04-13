from os import PathLike
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.modules.loss import _Loss
from torch.functional import F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._module import MultiClassModule
from .._loss import FocalLoss
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
        **kwargs: Any
    ) -> None:

        super().__init__(
            task_name, hyperparams, model_dir or Path(__file__).parent, **kwargs
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

    @classmethod
    def _create_loss(
        cls, data_loader: DataLoader[AsymmetricMultiClassDataset]
    ) -> _Loss:
        label_counts = cast(
            AsymmetricMultiClassDataset, data_loader.dataset
        ).label_counts
        total_count = label_counts.sum()
        n_classes = len(label_counts)
        weights = torch.tensor(
            total_count / (label_counts * n_classes), dtype=torch.float32
        )
        weights = torch.sqrt(weights)  # dampening
        weights = weights / weights[0]
        return FocalLoss(weights)

    def _create_optimizer(self, model: MultiClassModule) -> Optimizer:
        base_lr = self._params.learning_rate
        decay_factor = self._params.lr_decay_factor
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
    def _forward_pass(
        cls,
        model: Module,
        batch: tuple[dict, dict, torch.LongTensor],
        device: torch.device,
    ) -> tuple:
        x_fwd, x_rev, y = batch
        x_fwd = {k: v.to(device) for k, v in x_fwd.items()}
        x_rev = {k: v.to(device) for k, v in x_rev.items()}
        y = y.to(device)
        y_rev = y.clone()
        y_rev[y == 2] = (
            0  # when reversing the pairs, all data labeled initially with 2 is a non-match
        )
        cls_logits = model(**x_fwd)
        cls_logits_rev = model(**x_rev)
        return cls_logits, cls_logits_rev, y, y_rev

    def _compute_loss(
        self, epoch: int, loss_fn: _Loss, tensors: Iterable[Tensor]
    ) -> Any:
        cls_logits, cls_logits_rev, y, y_rev = tensors
        loss = loss_fn(cls_logits, y)

        valid_mask = y_rev < 2  # Filter out invalid targets (though none exist here)
        loss_rev = F.cross_entropy(cls_logits_rev[valid_mask], y_rev[valid_mask])
        class_2_penalty = F.softmax(cls_logits_rev, dim=1)[:, 2].mean()
        loss_rev += 2.0 * class_2_penalty  # Adjust weight (10.0) as needed
        return loss + loss_rev
