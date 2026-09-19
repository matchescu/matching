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


class BestByDevMetric:
    """Track the best dev metric and expose the epoch that achieved it."""

    def __init__(self, metric: str = "dev_mcc", min_delta: float = 0.0) -> None:
        self._metric = metric
        self._min_delta = min_delta
        self._best = -float("inf")
        self.best_epoch = -1

    def __call__(self, epoch: int, dev_metrics: dict) -> bool:
        value = float(dev_metrics[self._metric])
        if value > self._best + self._min_delta:
            self._best = value
            self.best_epoch = epoch
            return True
        return False


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
        self._selector = BestByDevMetric("dev_mcc", min_delta=0.0)
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

    @classmethod
    def _aggregate_predictions(
        cls, batch_results: list[tuple[torch.Tensor, ...]]
    ) -> tuple:
        y_pred, y_pred_rev, energy_fwd, energy_rev, y_true = zip(*batch_results)
        y_pred = torch.cat(y_pred).detach().cpu()
        y_pred_rev = torch.cat(y_pred_rev).detach().cpu()
        y_true = torch.cat(y_true).detach().cpu()
        energy = torch.stack(
            (torch.cat(energy_fwd), torch.cat(energy_rev)), dim=-1
        ).cpu()
        return y_pred, y_pred_rev, y_true, energy

    @staticmethod
    def _energy_metrics(
        energy: torch.Tensor, y_true: torch.Tensor, order_margin: float
    ) -> dict:
        energy_metrics: dict = {}
        for label in range(3):
            selected = energy[y_true == label]
            support = len(selected)
            means = (
                selected.mean(dim=0)
                if support
                else energy.new_full((2,), float("nan"))
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
        c2_support = energy_metrics["order_c2_support"]
        energy_metrics["order_c2_rev_above_margin"] = (
            (energy[y_true == 2, 1] > order_margin).float().mean().item()
            if c2_support
            else float("nan")
        )
        return energy_metrics

    @classmethod
    def _directional_metrics(
        cls, y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_rev: torch.Tensor
    ) -> dict:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_pred_rev = y_pred_rev.numpy()

        mcc_fwd = metrics.matthews_corrcoef(y_true, y_pred)

        y_true_rev = y_true.copy()
        y_true_rev[y_true == 2] = 0
        mcc_rev = metrics.matthews_corrcoef(y_true_rev, y_pred_rev)

        n = len(y_true)
        c2_support = int((y_true == 2).sum())
        c2_fwd_fn = (
            float(((y_true == 2) & (y_pred != 2)).sum()) / c2_support
            if c2_support
            else float("nan")
        )
        non_c2_support = n - c2_support
        c2_fwd_fp = (
            float(((y_true != 2) & (y_pred == 2)).sum()) / non_c2_support
            if non_c2_support
            else float("nan")
        )

        y1_mask = y_true == 1
        y1_support = int(y1_mask.sum())
        rev_fpr = (
            float((y1_mask & (y_pred_rev == 0)).sum()) / y1_support
            if y1_support
            else float("nan")
        )

        non_y1_mask = y_true != 1
        non_y1_support = int(non_y1_mask.sum())
        rev_fnr = (
            float((non_y1_mask & (y_pred_rev != 0)).sum()) / non_y1_support
            if non_y1_support
            else float("nan")
        )

        y2_mask = y_true == 2
        y2_support = int(y2_mask.sum())
        c2_rev_zero_acc = (
            float((y2_mask & (y_pred_rev == 0)).sum()) / y2_support
            if y2_support
            else float("nan")
        )
        c2_rev_ambiguous = (
            float((y2_mask & (y_pred_rev == 1)).sum()) / y2_support
            if y2_support
            else float("nan")
        )
        c2_rev_directed = (
            float((y2_mask & (y_pred_rev == 2)).sum()) / y2_support
            if y2_support
            else float("nan")
        )

        order_mask = y2_mask
        order_support = y2_support
        order_acc = (
            float(
                ((y_pred == 2) & (y_pred_rev == 0) & order_mask).sum()
            )
            / order_support
            if order_support
            else float("nan")
        )
        collapse = float((y_pred == y_pred_rev).sum()) / n

        return {
            "mcc": mcc_fwd,
            "mcc_rev": mcc_rev,
            "c2_fwd_fn": c2_fwd_fn,
            "c2_fwd_fp": c2_fwd_fp,
            "rev_fpr": rev_fpr,
            "rev_fnr": rev_fnr,
            "c2_rev_zero_acc": c2_rev_zero_acc,
            "c2_rev_ambiguous": c2_rev_ambiguous,
            "c2_rev_directed": c2_rev_directed,
            "order_acc": order_acc,
            "collapse": collapse,
        }

    @torch.no_grad()
    def _run_model(
        self,
        model: MultiClassModule,
        data: DataLoader[AsymmetricMultiClassDataset],
        best_config: dict | None = None,
    ) -> tuple[bool, dict]:
        batch_results = [
            (*self._interpret_result(model, batch_fwd, batch_rev), y_true)
            for batch_fwd, batch_rev, y_true in data
        ]
        y_pred, y_pred_rev, y_true, energy = self._aggregate_predictions(
            batch_results
        )

        directional = self._directional_metrics(y_true, y_pred, y_pred_rev)
        energy_metrics = self._energy_metrics(energy, y_true, model.order_margin)

        result = {**directional, **energy_metrics}

        if self._is_evaluating(best_config):
            best_config = best_config or {}
            best_config.update({f"test_{key}": value for key, value in result.items()})
            return True, best_config

        dev_result = {f"dev_{key}": value for key, value in result.items()}
        epoch = getattr(model, "_training_epoch", 1)
        success = self._selector(epoch, dev_result)
        return success, dev_result
