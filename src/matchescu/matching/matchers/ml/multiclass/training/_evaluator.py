import logging
from pathlib import Path

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._module import MultiClassModule
from ._config import CAPABILITY
from ._datasets import DittoDataset


class TrainingEvaluator(
    BaseEvaluator[MultiClassModule, DittoDataset], capability=CAPABILITY
):
    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[DittoDataset],
        test_data: DataLoader[DittoDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(task_name, xv_data, test_data, tb_log_dir, logger)
        self._best_xv = 0.0

    @torch.no_grad()
    def _run_model(
        self,
        model: MultiClassModule,
        data_loader: DataLoader[DittoDataset],
        best_config: dict | None = None,
    ) -> tuple[bool, dict]:
        batch_results = map(
            lambda b: (torch.argmax(model(b[0]), dim=1), b[1]), data_loader
        )
        y_pred, y_true = zip(*batch_results)
        y_pred = torch.cat(y_pred).detach().cpu().numpy()
        y_true = torch.cat(y_true).detach().cpu().numpy()

        mcc = metrics.matthews_corrcoef(y_true, y_pred)

        if self._is_evaluating(best_config):
            best_config.update({"test_mcc": mcc})
            return True, best_config
        else:
            if mcc > self._best_xv:
                self._best_xv = mcc
                return True, {"dev_mcc": mcc}
            else:
                return False, {}
