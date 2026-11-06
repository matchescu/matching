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
    THRESHOLD_KEY = "best_threshold"
    _THRESHOLD_GRID = np.arange(0.01, 1.0, 0.01)
    _PLATEAU_TOLERANCE = 0.005

    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[DittoDataset],
        test_data: DataLoader[DittoDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(task_name, xv_data, test_data, tb_log_dir, logger)
        self._best_xv_f1 = -1.0

    @staticmethod
    def __f1_curve(
        probabilities: np.ndarray,
        labels: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        predictions = (probabilities[:, None] > thresholds).astype(int)
        return np.array(
            [metrics.f1_score(labels, pred, zero_division=0) for pred in predictions.T],
            dtype=np.float64,
        )

    @staticmethod
    def __plateau_midpoint(
        thresholds: np.ndarray,
        f1_scores: np.ndarray,
        tolerance: float = _PLATEAU_TOLERANCE,
    ) -> tuple[float, float, float, float, float]:
        best_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_idx])
        good = f1_scores >= best_f1 - tolerance

        left = best_idx
        while left > 0 and good[left - 1]:
            left -= 1
        right = best_idx
        while right + 1 < len(good) and good[right + 1]:
            right += 1

        mid_idx = (left + right) // 2
        return (
            float(thresholds[mid_idx]),
            float(f1_scores[mid_idx]),
            float(thresholds[left]),
            float(thresholds[right]),
            best_f1,
        )

    @classmethod
    def __best_threshold(
        cls, probabilities: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float, dict]:
        thresholds = cls._THRESHOLD_GRID
        n_pos = int(labels.sum())
        n_neg = int(labels.size - n_pos)
        metadata = {"dev_pos": n_pos, "dev_neg": n_neg}

        if labels.size == 0:
            metadata["threshold_fallback"] = "empty"
            return 0.0, 0.5, metadata

        if n_pos == 0 or n_neg == 0:
            f1 = float(
                metrics.f1_score(
                    labels, (probabilities > 0.5).astype(int), zero_division=0
                )
            )
            metadata["threshold_fallback"] = "single_class"
            return f1, 0.5, metadata

        f1_scores = cls.__f1_curve(probabilities, labels, thresholds)
        threshold, f1, lo, hi, max_f1 = cls.__plateau_midpoint(thresholds, f1_scores)
        metadata.update(
            {
                "dev_auc": float(metrics.roc_auc_score(labels, probabilities)),
                "dev_f1_max": max_f1,
                "threshold_lo": lo,
                "threshold_hi": hi,
                "threshold_plateau_width": hi - lo,
            }
        )
        return f1, threshold, metadata

    @torch.no_grad()
    def _run_model(
        self,
        model: DittoModel,
        data_loader: DataLoader[DittoDataset],
        best_config: dict | None = None,
    ) -> tuple[bool, dict]:

        batch_results = ((torch.sigmoid(model(b[0])), b[1]) for b in data_loader)
        all_probs, all_y = zip(*batch_results)
        all_probs = torch.cat(all_probs).detach().cpu().numpy()
        all_y = torch.cat(all_y).detach().cpu().numpy()

        if self._is_evaluating(best_config):
            # evaluate the model
            threshold = float(best_config[self.THRESHOLD_KEY])
            pred = (all_probs > threshold).astype(int)
            f1 = metrics.f1_score(all_y, pred)
            best_config.update({"test_f1": f1, "test_threshold": threshold})
            return True, best_config

        # tune the model
        f1, threshold, metadata = TrainingEvaluator.__best_threshold(all_probs, all_y)
        self._log.info(
            "dev: f1=%.4f threshold=%.2f plateau=[%.2f, %.2f] auc=%.4f",
            f1,
            threshold,
            metadata.get("threshold_lo", threshold),
            metadata.get("threshold_hi", threshold),
            metadata.get("dev_auc", float("nan")),
        )
        if f1 <= self._best_xv_f1:
            return False, {}

        self._best_xv_f1 = f1
        return True, {"dev_f1": f1, self.THRESHOLD_KEY: threshold, **metadata}
