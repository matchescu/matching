import logging
from pathlib import Path

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._loss import order_energy
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
        self._best = -1.0

    def _interpret_result(
        self,
        model: MultiClassModule,
        batch_fwd: dict[str, torch.Tensor],
        batch_rev: dict[str, torch.Tensor],
    ):
        cls_logits, enc_a, enc_b = model(**batch_fwd, return_embeddings=True)
        cls_logits_rev = model(**batch_rev)
        cls_pred = torch.argmax(cls_logits, dim=-1)
        cls_pred_rev = torch.argmax(cls_logits_rev, dim=-1)
        self._log.info(
            "cls_logits: %s; cls_logits_rev: %s",
            cls_logits.softmax(dim=1).mean(dim=0),
            cls_logits_rev.softmax(dim=1).mean(dim=0),
        )
        return (
            cls_pred,
            cls_pred_rev,
            order_energy(enc_a, enc_b),
            order_energy(enc_b, enc_a),
        )

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
        y_pred, y_pred_rev, energy_fwd, energy_rev, y_true = zip(*batch_results)
        y_pred = torch.cat(y_pred).detach().cpu().numpy()
        y_pred_rev = torch.cat(y_pred_rev).detach().cpu().numpy()
        y_true = torch.cat(y_true).detach().cpu()
        energy = torch.stack(
            (torch.cat(energy_fwd), torch.cat(energy_rev)), dim=-1
        ).cpu()
        energy_metrics = {}
        for label in range(3):
            selected = energy[y_true == label]
            support = len(selected)
            means = (
                selected.mean(dim=0) if support else energy.new_full((2,), float("nan"))
            )
            stds = selected.std(dim=0, correction=0) if support else means
            energy_metrics.update(
                {
                    f"order_c{label}_support": support,
                    f"order_c{label}_fwd_mean": means[0].item(),
                    f"order_c{label}_fwd_std": stds[0].item(),
                    f"order_c{label}_rev_mean": means[1].item(),
                    f"order_c{label}_rev_std": stds[1].item(),
                }
            )
        energy_metrics["order_c2_rev_above_margin"] = (
            (energy[y_true == 2, 1] > model.order_margin).float().mean().item()
        )
        y_true = y_true.numpy()
        y_true_rev = y_true.copy()
        y_true_rev[y_true == 2] = 0
        avg_loss = float(best_config.get("average_loss", 1))

        mcc_normal = metrics.matthews_corrcoef(y_true, y_pred)
        mcc_rev = metrics.matthews_corrcoef(y_true_rev, y_pred_rev)
        mcc = (mcc_normal + mcc_rev) / 2

        n = len(y_true)
        c2_fwd_fn = float(((y_true == 2) & (y_pred != 2)).sum()) / n
        c2_fwd_fp = float(((y_true != 2) & (y_pred == 2)).sum()) / n
        c2_rev_fp = float((y_pred_rev == 2).sum()) / n

        if self._is_evaluating(best_config):
            best_config.update(
                {
                    "test_mcc": mcc,
                    "test_c2_fwd_fn": c2_fwd_fn,
                    "test_c2_fwd_fp": c2_fwd_fp,
                    "test_c2_rev_fp": c2_rev_fp,
                    **{f"test_{key}": value for key, value in energy_metrics.items()},
                }
            )
            return True, best_config
        else:
            success = False
            current = mcc / avg_loss
            if current > self._best:
                self._best = current
                success = True
            return success, {
                "dev_mcc": mcc,
                "dev_c2_fwd_fn": c2_fwd_fn,
                "dev_c2_fwd_fp": c2_fwd_fp,
                "dev_c2_rev_fp": c2_rev_fp,
                **{f"dev_{key}": value for key, value in energy_metrics.items()},
            }
