import itertools
import logging

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.ditto._ditto_dataset import DittoDataset
from matchescu.matching.matchers.ml.ditto._ditto_module import DittoModel


class DittoTrainingEvaluator:
    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[DittoDataset],
        test_data: DataLoader[DittoDataset],
        logger: logging.Logger | None = None,
    ) -> None:
        self._task = task_name
        self._xv_data = xv_data
        self._test_data = test_data
        self._log = (logger or logging.getLogger(self.__class__.__name__)).getChild(
            self._task
        )

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

    @staticmethod
    @torch.no_grad()
    def _evaluate_model(
        model: DittoModel, data_loader: DataLoader, threshold: float | None = None
    ):
        model.eval()

        batch_results = map(lambda b: (torch.sigmoid(model(b[0])), b[1]), data_loader)
        all_probs, all_y = zip(*batch_results)
        all_probs = torch.cat(all_probs).detach().cpu().numpy()
        all_y = torch.cat(all_y).detach().cpu().numpy()

        if threshold is not None:
            pred = (all_probs > threshold).astype(int)
            f1 = metrics.f1_score(all_y, pred)
            return f1, threshold
        return DittoTrainingEvaluator.__best_threshold(all_probs, all_y)

    def __call__(self, model: DittoModel) -> tuple[float, float, float]:
        self._log.info("evaluating on cross-validation set")
        xv_f1, best_xv_threshold = self._evaluate_model(model, self._xv_data)
        self._log.info(
            "X validation F1=%.4f, best threshold=%.2f",
            xv_f1,
            best_xv_threshold,
        )
        test_f1, _ = self._evaluate_model(
            model, self._test_data, threshold=best_xv_threshold
        )
        self._log.info("test F1=%.4f", test_f1)
        return xv_f1, test_f1, best_xv_threshold
