import logging
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from matchescu.matching.matchers.ml.ditto.training._datasets import DittoDataset


class TrainingEvaluator(AbstractContextManager):
    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[DittoDataset],
        test_data: DataLoader[DittoDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._task = task_name
        self._xv_data = xv_data
        self._test_data = test_data
        self._best_test_f1 = 0.0
        self._best_xv_f1 = 0.0
        self._tb_log_dir = tb_log_dir.absolute()
        self._summary_writer = SummaryWriter(log_dir=str(self._tb_log_dir))
        self._log = (logger or logging.getLogger(self.__class__.__name__)).getChild(
            self._task
        )

    @property
    def summary_writer(self):
        return self._summary_writer

    @staticmethod
    @torch.no_grad()
    def _evaluate_model(model: nn.Module, data_loader: DataLoader):
        model.eval()
        y_pred = []
        y_true = []

        for batch in data_loader:
            left_attrs = batch["left_attrs"]
            right_attrs = batch["right_attrs"]
            labels = batch["label"].long()

            logits = model(left_attrs, right_attrs)
            y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())
            y_true.extend(labels.detach().cpu().numpy())

        return f1_score(y_true, y_pred)

    def __call__(self, model: nn.Module, epoch: int) -> tuple[bool, float]:
        self._log.info("evaluating on cross-validation set")
        xv_f1 = self._evaluate_model(model, self._xv_data)
        self._log.info("dev F1=%.4f", xv_f1)
        test_f1 = self._evaluate_model(model, self._test_data)
        self._log.info("test F1=%.4f", test_f1)
        self._summary_writer.add_scalars(
            self._task, {"dev F1": xv_f1, "test F1": test_f1}, epoch
        )

        found_new_best = False
        if xv_f1 > self._best_xv_f1:
            self._log.info("found new best F1. saving checkpoint")
            self._best_xv_f1 = xv_f1
            self._best_test_f1 = test_f1

            self._log.info(
                "xv_f1=%.4f, best_xv_f1=%.4f, test_f1=%.4f, best_test_f1=%.4f",
                xv_f1,
                self._best_xv_f1,
                test_f1,
                self._best_test_f1,
            )
            found_new_best = True

        return found_new_best, xv_f1

    def __enter__(self) -> "TrainingEvaluator":
        if self._summary_writer is not None:
            return self
        self._summary_writer = SummaryWriter(log_dir=str(self._tb_log_dir))
        return self

    def __exit__(self, exc_type, exc_value, traceback, /):
        if self._summary_writer is None:
            return

        self._summary_writer.flush()
        self._summary_writer.close()
        self._summary_writer = None
