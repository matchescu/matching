import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from matchescu.matching.matchers.ml.training import BaseEvaluator
from matchescu.matching.matchers.ml.deepmatcher import DeepMatcherModule
from matchescu.matching.matchers.ml.deepmatcher.training._dataset import (
    DeepMatcherDataset,
)


class TrainingEvaluator(
    BaseEvaluator[DeepMatcherModule, DeepMatcherDataset], capability="deepmatcher"
):
    def __init__(
        self,
        task_name: str,
        dev_data: DataLoader[DeepMatcherDataset],
        test_data: DataLoader[DeepMatcherDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(task_name, dev_data, test_data, tb_log_dir, logger)
        self._best_xv_f1 = 0.0

    def _run_model(
        self,
        model: DeepMatcherModule,
        data: DataLoader[DeepMatcherDataset],
        **kwargs: Any
    ) -> tuple[bool, dict]:
        y_pred = []
        y_true = []

        for batch in data:
            left_attrs = batch["left_attrs"]
            right_attrs = batch["right_attrs"]
            labels = batch["label"].long()

            logits = model(left_attrs, right_attrs)
            y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())
            y_true.extend(labels.detach().cpu().numpy())

        f1 = f1_score(y_true, y_pred)

        if self._is_evaluating(**kwargs):
            kwargs.update({"test_f1": f1})
            return True, kwargs
        else:
            if f1 > self._best_xv_f1:
                return True, {"dev_f1": f1}
            else:
                return False, {}
