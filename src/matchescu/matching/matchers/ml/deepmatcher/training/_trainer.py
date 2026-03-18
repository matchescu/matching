from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.deepmatcher.training._dataset import (
    DeepMatcherDataset,
)
from matchescu.matching.matchers.ml.deepmatcher.training._evaluator import (
    TrainingEvaluator,
)


class DeepMatcherTrainer:
    def __init__(
        self,
        task_name: str,
        model_dir: str | PathLike | None = None,
        loss_fn: _Loss | None = None,
        **kwargs: Any,
    ) -> None:
        self._task = task_name
        self._model_dir = Path(model_dir) if model_dir else Path(__file__).parent
        self._log = cast(
            Logger, kwargs.get("logger", getLogger(self.__class__.__name__))
        ).getChild(self._task)
        self._epochs = int(kwargs.get("epochs", 20))
        self._learning_rate = float(kwargs.get("learning_rate", 3e-5))
        self._loss = loss_fn or nn.CrossEntropyLoss()

    def __get_device(self):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            if torch.backends.mps.is_built():
                self._log.info("MPS built, but not available.")
            else:
                self._log.info("Not Mac, nor CUDA.")
        return device

    def _train_one_epoch(
        self,
        epoch: int,
        device: torch.device,
        model: torch.nn.Module,
        train_iter: DataLoader,
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
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)
                optimizer.zero_grad()

                prediction = model(**inputs)
                loss = loss_fn(prediction, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                total_loss += step_loss
                batch_loss += step_loss
                batch_no = i + 1
                if batch_no % 10 == 0:
                    batch_loss = batch_loss / 10
                    fmt = f"batch {batch_no}: avg loss over last 10 batches=%.4f"
                    self._log.info(fmt, batch_loss)
                del loss
        finally:
            model.train(False)

        avg_loss = total_loss / batch_no if batch_no > 0 else 0
        self._log.info("epoch %d: avg loss=%.4f", epoch, avg_loss)
        return {"Average Loss": avg_loss}

    def run_training(
        self,
        model: nn.Module,
        training_data: DataLoader[DeepMatcherDataset],
        evaluator: TrainingEvaluator,
        save_model: bool = False,
    ):
        device = self.__get_device()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate)
        ds = cast(DeepMatcherDataset, training_data.dataset)
        batch_count = int(len(ds) // training_data.batch_size)
        num_steps = batch_count * self._epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_steps
        )

        for epoch in range(1, self._epochs + 1):
            self._log.info("epoch %d - train start", epoch)
            try:
                self._train_one_epoch(
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

            is_new_best, dev_f1 = evaluator(model, epoch)
            if is_new_best:
                self._save_checkpoint(epoch, model, optimizer, scheduler)

    def _save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ):
        task_model_dir = self._model_dir / self._task
        task_model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = task_model_dir / "model.pt"
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, ckpt_path)
