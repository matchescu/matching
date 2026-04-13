import logging
from pathlib import Path

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._module import MultiClassModule
from ._config import CAPABILITY
from ._datasets import AsymmetricMultiClassDataset


class TrainingEvaluator(
    BaseEvaluator[MultiClassModule, AsymmetricMultiClassDataset], capability=CAPABILITY
):
    def __init__(
        self,
        task_name: str,
        xv_data: DataLoader[AsymmetricMultiClassDataset],
        test_data: DataLoader[AsymmetricMultiClassDataset],
        tb_log_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(task_name, xv_data, test_data, tb_log_dir, logger)
        self._best_mcc = -1.0

    def _interpret_result(
        self,
        model: MultiClassModule,
        batch_fwd: dict[str, torch.Tensor],
        batch_rev: dict[str, torch.Tensor],
    ):
        cls_logits = model(**batch_fwd)
        cls_logits_rev = model(**batch_rev)
        cls_pred = torch.argmax(cls_logits, dim=-1)
        cls_pred_rev = torch.argmax(cls_logits_rev, dim=-1)
        self._log.info(
            "cls_logits: %s; cls_logits_rev: %s",
            cls_logits.softmax(dim=1).mean(dim=0),
            cls_logits_rev.softmax(dim=1).mean(dim=0),
        )
        return cls_pred, cls_pred_rev

    @torch.no_grad()
    def _run_model(
        self,
        model: MultiClassModule,
        data_loader: DataLoader[AsymmetricMultiClassDataset],
        best_config: dict | None = None,
    ) -> tuple[bool, dict]:
        batch_results = [
            (*self._interpret_result(model, batch_fwd, batch_rev), y_true)
            for batch_fwd, batch_rev, y_true in data_loader
        ]
        y_pred, y_pred_rev, y_true = zip(*batch_results)
        y_pred = torch.cat(y_pred).detach().cpu().numpy()
        y_pred_rev = torch.cat(y_pred_rev).detach().cpu().numpy()
        y_true = torch.cat(y_true).detach().cpu().numpy()
        y_true_rev = y_true.copy()
        y_true_rev[y_true == 2] = 0

        mcc_normal = metrics.matthews_corrcoef(y_true, y_pred)
        mcc_rev = metrics.matthews_corrcoef(y_true_rev, y_pred_rev)
        mcc = (mcc_normal + mcc_rev) / 2

        if self._is_evaluating(best_config):
            best_config.update({"test_mcc": mcc})
            return True, best_config
        else:
            success = False
            if mcc > self._best_mcc:
                self._best_mcc = mcc
                success = True
            return success, {"dev_mcc": mcc}
