from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._loss import FocalLoss
from .._module import MultiClassModule
from .._params import MultiClassTrainingParams
from .._types import LossType
from ._config import CAPABILITY
from ._datasets import AsymmetricMultiClassDataset


class MultiClassTrainer(
    BaseTrainer[
        MultiClassModule, MultiClassTrainingParams, AsymmetricMultiClassDataset
    ],
    capability=CAPABILITY,
):
    _FOCAL_GAMMA = 2.0
    _DIRECTIONAL_MARGIN = 2.0
    hyperparams_schema = MultiClassTrainingParams

    def __init__(
        self,
        task_name: str,
        hyperparams: MultiClassTrainingParams,
        model_dir: str | PathLike | None = None,
        **kwargs: Any,
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

    @staticmethod
    def _compute_weights(label_counts: np.ndarray) -> torch.Tensor:
        c0, c1, c2 = torch.as_tensor(label_counts, dtype=torch.float)
        total_positive = 2 * c1 + c2
        total_negative = 2 * c0 + c2
        weights = torch.stack([1.0 / total_negative, 1.0 / total_positive])
        weights = weights * (2.0 / weights.sum())
        return weights

    def _create_loss(
        self, data_loader: DataLoader[AsymmetricMultiClassDataset]
    ) -> _Loss:
        lc = cast(AsymmetricMultiClassDataset, data_loader.dataset).label_counts
        weights = self._compute_weights(lc)

        match self._params.loss_type:
            case LossType.WEIGHTED_CE:
                return FocalLoss(weights, gamma=0.0)
            case LossType.FOCAL:
                return FocalLoss(weights, gamma=self._FOCAL_GAMMA)

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

        if model.cross_attention is not None:
            param_groups.extend(
                self._get_decay_model_params(
                    model.cross_attention, base_lr / decay_factor, weight_decay
                )
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
        x_fwd, x_rev, y_fwd = batch
        x_fwd = {k: v.to(device) for k, v in x_fwd.items()}
        x_rev = {k: v.to(device) for k, v in x_rev.items()}
        y_fwd = y_fwd.to(device)
        y_rev = torch.where(y_fwd == 2, torch.zeros_like(y_fwd), y_fwd).to(device)

        logits_fwd = model(**x_fwd)
        logits_rev = model(**x_rev)
        bits_fwd = torch.stack(((y_fwd > 0), (y_fwd == 1)), dim=1).long().to(device)
        bits_rev = torch.stack(((y_rev > 0), (y_rev == 1)), dim=1).long().to(device)

        return logits_fwd, logits_rev, bits_fwd, bits_rev

    def _compute_loss(
        self, epoch: int, loss_fn: _Loss, tensors: Iterable[Tensor]
    ) -> Any:
        logits_fwd, logits_rev, bits_fwd, bits_rev = tensors
        # rearrange logits as (N, bit, class)
        z_fwd = logits_fwd.float().view(-1, 2, 2)
        z_rev = logits_rev.float().view(-1, 2, 2)
        bits_fwd = bits_fwd.long()
        bits_rev = bits_rev.long()

        # compute loss on 2 separate objectives in fwd and rev orders
        loss_fwd = loss_fn(z_fwd[:, 0], bits_fwd[:, 0]) + loss_fn(
            z_fwd[:, 1], bits_fwd[:, 1]
        )
        loss_rev = loss_fn(z_rev[:, 0], bits_rev[:, 0]) + loss_fn(
            z_rev[:, 1], bits_rev[:, 1]
        )
        co_loss = self._params.cross_order_weight * self._cross_order_consistency(
            z_fwd, z_rev
        )
        return loss_fwd + loss_rev + co_loss

    @staticmethod
    def _cross_order_consistency(z_fwd: Tensor, z_rev: Tensor) -> Tensor:
        # get the {0, 1} predictions in each direction (actual p and q values)
        fwd_pos = z_fwd.softmax(dim=-1)[..., 1]
        rev_pos = z_rev.softmax(dim=-1)[..., 1]
        p_fwd, q_fwd = fwd_pos.unbind(dim=1)
        p_rev, q_rev = rev_pos.unbind(dim=1)
        # Mean-squared disagreement between the two estimates of each direction.
        return (
            (q_rev - p_fwd.detach()).pow(2) + (p_rev - q_fwd.detach()).pow(2)
        ).mean()
