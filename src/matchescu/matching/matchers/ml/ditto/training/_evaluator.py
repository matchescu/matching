import itertools
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._module import DittoModel
from ._config import CAPABILITY
from ._datasets import DittoDataset


class TrainingEvaluator(BaseEvaluator[DittoModel, DittoDataset], capability=CAPABILITY):
    _BEST_THRESHOLD_KEY = "best_threshold"

    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[DittoDataset],
        test_data: DataLoader[DittoDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(task_name, xv_data, test_data, tb_log_dir, logger)
        self._best_xv_f1 = 0.0

    @staticmethod
    def __best_threshold(
        probabilities: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        thresholds = np.arange(0.0, 1.0, 0.05)
        predictions = (probabilities[:, None] > thresholds).astype(int)
        f1_scores = np.fromiter(
            itertools.starmap(
                metrics.f1_score, zip(itertools.repeat(labels), predictions.T)
            ),
            dtype=np.float32,
        )
        best_idx = np.argmax(f1_scores)
        return float(f1_scores[best_idx]), float(thresholds[best_idx])

    @torch.no_grad()
    def _run_model(
        self,
        model: DittoModel,
        data_loader: DataLoader[DittoDataset],
        best_config: dict | None = None,
    ) -> tuple[bool, dict]:

        batch_results = map(lambda b: (torch.sigmoid(model(b[0])), b[1]), data_loader)
        all_probs, all_y = zip(*batch_results)
        all_probs = torch.cat(all_probs).detach().cpu().numpy()
        all_y = torch.cat(all_y).detach().cpu().numpy()

        if self._is_evaluating(best_config):
            # evaluate the model
            threshold = float(best_config[self._BEST_THRESHOLD_KEY])
            pred = (all_probs > threshold).astype(int)
            f1 = metrics.f1_score(all_y, pred)
            best_config.update({"test_f1": f1, "test_threshold": threshold})
            return True, best_config

        # tune the model
        f1, threshold = TrainingEvaluator.__best_threshold(all_probs, all_y)
        if f1 <= self._best_xv_f1:
            return False, {}

        self._best_xv_f1 = f1
        return True, {"dev_f1": f1, self._BEST_THRESHOLD_KEY: threshold}
