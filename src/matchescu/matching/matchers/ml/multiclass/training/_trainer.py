from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.training import BaseTrainer

from .._loss import FocalLoss, directional_margin_loss, order_loss
from .._module import MultiClassModule
from .._params import MultiClassTrainingParams
from .._types import LossType
from ._config import CAPABILITY
from ._datasets import AsymmetricMultiClassDataset
from ._evaluator import TrainingEvaluator


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

    def _create_loss(
        self, data_loader: DataLoader[AsymmetricMultiClassDataset]
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
        match self._params.loss_type:
            case LossType.WEIGHTED_CE:
                return FocalLoss(weights, gamma=0.0)
            case LossType.FOCAL:
                return FocalLoss(weights, gamma=self._params.focal_gamma)

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
    ) -> tuple[Tensor, ...]:
        x_fwd, x_rev, y = batch
        x_fwd = {k: v.to(device) for k, v in x_fwd.items()}
        x_rev = {k: v.to(device) for k, v in x_rev.items()}
        y = y.to(device)
        y_rev = y.clone()
        y_rev[y == 2] = (
            0  # when reversing the pairs, all data labeled initially with 2 is a non-match
        )
        cls_logits, enc_a, enc_b = model(**x_fwd, return_embeddings=True)
        cls_logits_rev = model(**x_rev)
        return cls_logits, cls_logits_rev, y, y_rev, enc_a, enc_b

    def _compute_loss(
        self, epoch: int, loss_fn: _Loss, tensors: Iterable[Tensor]
    ) -> dict[str, Tensor]:
        cls_logits, cls_logits_rev, y, y_rev, enc_a, enc_b = tensors
        cls_logits, cls_logits_rev, y, y_rev = (
            cls_logits.float(),
            cls_logits_rev.float(),
            y.long(),
            y_rev.long(),
        )

        loss_fwd = loss_fn(cls_logits, y)
        valid_mask = y_rev < 2  # Filter out invalid targets (though none exist here)
        loss_rev = loss_fn(cls_logits_rev[valid_mask], y_rev[valid_mask])
        loss_dir = directional_margin_loss(
            cls_logits, cls_logits_rev, y, self._params.dir_margin
        )
        loss_order = order_loss(enc_a, enc_b, y, self._params.order_margin)
        return {
            "total": loss_fwd
            + loss_rev
            + self._params.dir_margin_weight * loss_dir
            + self._params.order_loss_weight * loss_order,
            "loss_fwd": loss_fwd,
            "loss_rev": loss_rev,
            "loss_dir": loss_dir,
            "loss_order": loss_order,
        }

    def _train_one_epoch(
        self,
        epoch: int,
        device: torch.device,
        model: torch.nn.Module,
        train_iter: DataLoader[AsymmetricMultiClassDataset],
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> dict:
        metrics = super()._train_one_epoch(
            epoch, device, model, train_iter, optimizer, scheduler
        )
        train_dataset = getattr(train_iter, "dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_data_loader"):
            train_loader = train_dataset.get_data_loader(
                self._params.batch_size, shuffle=False, sampler=None
            )
            train_eval = TrainingEvaluator._measure(
                model, train_loader, self._params.order_margin, device
            )
            metrics.update({f"train_{key}": value for key, value in train_eval.items()})
        return metrics

    def run_training(
        self,
        model: MultiClassModule,
        training_data: DataLoader[AsymmetricMultiClassDataset],
        evaluator: TrainingEvaluator | None = None,
        save_model: bool = False,
    ) -> None:
        if evaluator is not None and not save_model:
            raise ValueError(
                "multiclass training requires save_model=True when an evaluator is supplied"
            )
        if evaluator is not None:
            evaluator.reset_selector()

        super().run_training(model, training_data, evaluator, save_model)

        if evaluator is None or not save_model:
            return

        ckpt_path = self._model_dir / self._task / "model.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"no best checkpoint found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_epoch = evaluator.best_epoch
        if ckpt.get("epoch") != best_epoch:
            raise RuntimeError(
                f"checkpoint epoch {ckpt.get('epoch')} does not match best epoch {best_epoch}"
            )

        model.load_state_dict(ckpt["model"])
        optimizer = self._create_optimizer(model)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler = self._create_scheduler(training_data.dataset, optimizer)
        scheduler.load_state_dict(ckpt["scheduler"])

        model.eval()
        best_config = dict(ckpt["additional_info"]["best_config"])
        best_config["is_evaluating"] = True
        ok, final_config = evaluator._run_model(
            model, evaluator._test_data, best_config
        )
        if not ok:
            raise RuntimeError("failed to evaluate selected checkpoint on test data")

        self._log.info(
            "selected epoch %d: %s", best_epoch, evaluator._repr_config(final_config)
        )
        ckpt["additional_info"]["best_config"] = final_config
        torch.save(ckpt, ckpt_path)
