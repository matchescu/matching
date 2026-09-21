import logging
from pathlib import Path

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._decoder import decode_logits
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
        self._best = (0, 0)

    def _interpret_result(
        self,
        model: MultiClassModule,
        batch_fwd: dict[str, torch.Tensor],
        batch_rev: dict[str, torch.Tensor],
    ):
        cls_logits = model(**batch_fwd)
        cls_logits_rev = model(**batch_rev)
        cls_pred = decode_logits(cls_logits)
        cls_pred_rev = decode_logits(cls_logits_rev)
        self._log.info(
            "pred dist fwd: %s; pred dist rev: %s",
            torch.bincount(cls_pred, minlength=3).float() / cls_pred.numel(),
            torch.bincount(cls_pred_rev, minlength=3).float() / cls_pred_rev.numel(),
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
        y_pred = torch.cat(y_pred).detach().cpu()
        y_pred_rev = torch.cat(y_pred_rev).detach().cpu()
        y_true = torch.cat(y_true).detach().cpu()
        y_true_rev = y_true.clone()
        y_true_rev[y_true == 2] = 0

        mcc_fwd = metrics.matthews_corrcoef(y_true.numpy(), y_pred.numpy())
        mcc_rev = metrics.matthews_corrcoef(y_true_rev.numpy(), y_pred_rev.numpy())
        bit_p_acc = float(((y_true > 0) == (y_pred > 0)).float().mean())
        bit_q_acc = float(((y_true == 1) == (y_pred == 1))[y_true > 0].float().mean())

        if self._is_evaluating(best_config):
            best_config.update(
                {
                    "test_mcc_fwd": mcc_fwd,
                    "test_mcc_rev": mcc_rev,
                    "test_bit_p_acc": bit_p_acc,
                    "test_bit_q_acc": bit_q_acc,
                }
            )
            return True, best_config
        else:
            success = False
            if (mcc_fwd, mcc_rev) > self._best:
                self._best = (mcc_fwd, mcc_rev)
                success = True
            return success, {
                "dev_mcc_fwd": mcc_fwd,
                "dev_mcc_rev": mcc_rev,
                "dev_bit_p_acc": bit_p_acc,
                "dev_bit_q_acc": bit_q_acc,
            }
