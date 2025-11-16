from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from matchescu.matching.matchers.ml.ditto._ditto_module import DittoModel
from matchescu.matching.matchers.ml.ditto.training._datasets import DittoDataset
from matchescu.matching.matchers.ml.ditto.training._evaluator import TrainingEvaluator


class DittoTrainer:
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
        self._frozen_layer_count = int(kwargs.get("frozen_layer_count", 0))
        self._loss = loss_fn or BCEWithLogitsLoss()

    def __get_device(self):
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
            loss_fn = self._loss.to(device)  # CrossEntropy
            model.to(device)
            model.train(True)
            batch_loss = 0.0
            for i, batch in enumerate(train_iter):
                device_batch = tuple(item.to(device) for item in batch)
                optimizer.zero_grad()

                if len(device_batch) == 2:
                    x, y = device_batch
                    prediction = model(x.to(device))
                else:
                    x1, x2, y = device_batch
                    prediction = model(x1.to(device), x2.to(device))

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
        model: DittoModel,
        training_data: DataLoader[DittoDataset],
        evaluator: TrainingEvaluator,
        save_model: bool = False,
        summary_writer: SummaryWriter | None = None,
    ):
        device = self.__get_device()
        model = model.with_frozen_bert_layers(self._frozen_layer_count)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate)
        total_batches = (
            len(cast(DittoDataset, training_data.dataset)) // training_data.batch_size
        )
        num_steps = total_batches * self._epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_steps
        )

        for epoch in range(1, self._epochs + 1):
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

            is_new_best, threshold = evaluator(model, train_metrics, epoch)

            # TODO: replace conditional with evaluation.is_new_highscore
            if is_new_best:
                self._save_checkpoint(epoch, model, optimizer, scheduler, threshold)

    def _save_checkpoint(
        self,
        epoch: int,
        model: DittoModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        threshold: float,
    ):
        task_model_dir = self._model_dir / self._task
        task_model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = task_model_dir / "model.pt"
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
        }
        torch.save(ckpt, ckpt_path)
