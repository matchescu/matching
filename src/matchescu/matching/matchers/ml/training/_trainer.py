from abc import ABC, abstractmethod
from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import ClassVar, Type, Any, Generic, cast

import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.core import (
    AdditionalModelInfo,
    ModelTrainingParams,
    TModel,
    TParams,
    TDataset,
)
from ._evaluator import BaseEvaluator
from ._registry import CapabilityRegistry


class BaseTrainer(ABC, Generic[TModel, TParams, TDataset]):
    """Subclasses of this class train matcher models.

    They can also be configured via config files if they advertise a ``capability``.
    The capability of each trainer tells the configuration system what types of
    hyperparameters they support for training and the kinds of models they can
    handle.
    """

    capability: ClassVar[str] = ""
    hyperparams_schema: ClassVar[Type[ModelTrainingParams]] = ModelTrainingParams

    def __init__(
        self,
        task: str,
        hyper_params: TParams,
        model_dir: str | PathLike,
        loss_fn: _Loss | None = None,
        **kwargs: Any,
    ) -> None:
        self._task = task
        self._params = hyper_params
        self._model_dir = Path(model_dir)
        self._loss = loss_fn
        self._log = cast(
            Logger, kwargs.get("logger", getLogger(self.__class__.__name__))
        ).getChild(self._task)

    def __init_subclass__(cls, capability: str = "", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if capability:
            cls.capability = capability
            CapabilityRegistry.register_trainer(capability, cls)

    def _get_device(self):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            if torch.backends.mps.is_built():
                self._log.info("MPS built, but not available.")
            else:
                self._log.info("Not Mac, nor CUDA.")
        return device

    def _setup_model(self, model: TModel) -> TModel:
        return model

    def _create_optimizer(self, model: TModel):
        return torch.optim.AdamW(model.parameters(), lr=self._params.learning_rate)

    def _create_scheduler(self, dataset: TDataset, optimizer: Optimizer):
        remainder = int(len(dataset) % self._params.batch_size > 0)
        total_batches = len(dataset) // self._params.batch_size + remainder
        num_steps = total_batches * self._params.epochs

        return OneCycleLR(
            optimizer,
            self._params.learning_rate,
            total_steps=num_steps,
            epochs=self._params.epochs,
            anneal_strategy="cos",
        )

    @classmethod
    @abstractmethod
    def _forward_pass(cls, model: TModel, batch: Any, device: torch.device) -> tuple:
        raise NotImplementedError

    def _train_one_epoch(
        self,
        epoch: int,
        device: torch.device,
        model: torch.nn.Module,
        train_iter: DataLoader[TDataset],
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ):
        total_loss = 0.0
        batch_no = 0

        try:
            loss_fn = self._loss.to(device)
            model.to(device)
            model.train(True)
            batch_loss = 0.0
            for i, batch in enumerate(train_iter):
                optimizer.zero_grad()
                prediction, y = self._forward_pass(model, batch, device)

                loss = loss_fn(prediction, y.to(device).float())

                loss.backward()
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                total_loss += step_loss
                batch_loss += step_loss
                batch_no = i + 1
                if batch_no % 10 == 0:
                    batch_loss = batch_loss / 10
                    self._log.info(
                        "batch %d: avg loss over last 10 batches=%.4f",
                        batch_no,
                        batch_loss,
                    )
                del loss
        finally:
            model.train(False)

        avg_loss = total_loss / batch_no if batch_no > 0 else 0
        self._log.info("epoch %d: avg loss=%.4f", epoch, avg_loss)
        return {"average_loss": avg_loss}

    def run_training(
        self,
        model: TModel,
        training_data: DataLoader[TDataset],
        evaluator: BaseEvaluator[TModel, TDataset],
        save_model: bool = False,
    ):
        device = self._get_device()
        model = self._setup_model(model)
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(training_data.dataset, optimizer)

        for epoch in range(1, self._params.epochs + 1):
            self._log.info("epoch %d - train start", epoch)
            try:
                train_metrics = self._train_one_epoch(
                    epoch,
                    device,
                    model,
                    training_data,
                    optimizer,
                    scheduler,
                )
            finally:
                self._log.info("epoch %d - train end", epoch)

            if evaluator is None or not save_model:
                continue

            has_new_best, best_config = evaluator(model, train_metrics, epoch)
            if has_new_best:
                self._save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    AdditionalModelInfo(
                        hyperparameters=self._params, best_config=best_config
                    ),
                )

    def _save_checkpoint(
        self,
        epoch: int,
        model: TModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        additional_info: AdditionalModelInfo[TParams],
    ):
        task_model_dir = self._model_dir / self._task
        task_model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = task_model_dir / "model.pt"
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "additional_info": additional_info.model_dump(),
        }
        torch.save(ckpt, ckpt_path)
        self._log.info("saved checkpoint to %s", ckpt_path)
