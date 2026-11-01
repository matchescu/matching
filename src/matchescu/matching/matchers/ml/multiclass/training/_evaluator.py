import logging
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor
from torch.utils.data import DataLoader

from matchescu.matching.matchers.ml.training import BaseEvaluator

from .._decoder import decode_logits
from .._module import MultiClassModule
from ._config import CAPABILITY
from ._datasets import AsymmetricMultiClassDataset

_NAN = float("nan")


@dataclass(frozen=True)
class BatchOutputs:
    y_pred: Tensor
    y_pred_rev: Tensor
    pq_fwd: Tensor
    pq_rev: Tensor
    pp_fwd: Tensor
    pp_rev: Tensor
    p_dir: Tensor
    y_true: Tensor

    @classmethod
    def concat(cls, batches: list["BatchOutputs"]) -> "BatchOutputs":
        return cls(
            *(
                torch.cat([getattr(batch, field.name) for batch in batches])
                .detach()
                .cpu()
                for field in fields(cls)
            )
        )


def _bit_prob(logits: Tensor, bit: int) -> Tensor:
    return logits.float().view(-1, 2, 2)[:, bit].softmax(-1)[:, 1]


def _bit_margin(logits: Tensor, bit: int) -> Tensor:
    logits = logits.float().view(-1, 2, 2)[:, bit]
    return logits[:, 1] - logits[:, 0]


def _mean(values: Tensor) -> float:
    return float(values.float().mean()) if values.numel() else _NAN


def _has_both_classes(y_bin: np.ndarray) -> bool:
    return y_bin.size > 0 and y_bin.min() != y_bin.max()


def _safe_auc(y_bin: np.ndarray, scores: np.ndarray) -> float:
    if not _has_both_classes(y_bin):
        return _NAN
    return float(metrics.roc_auc_score(y_bin, scores))


def _best_threshold_acc(y_bin: np.ndarray, scores: np.ndarray) -> float:
    if not _has_both_classes(y_bin):
        return _NAN
    fpr, tpr, _ = metrics.roc_curve(y_bin, scores)
    n_pos = y_bin.sum()
    n_neg = y_bin.size - n_pos
    return float(((tpr * n_pos + (1 - fpr) * n_neg) / y_bin.size).max())


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2 * left * right / (left + right)


def q_diagnostics(
    pq_fwd: Tensor, pq_rev: Tensor, pp_rev: Tensor, y_true: Tensor
) -> dict[str, float]:
    y1, y2 = y_true == 1, y_true == 2
    match = y1 | y2
    n1, n2 = int(y1.sum()), int(y2.sum())
    yes_fwd = pq_fwd > 0.5
    acc_y1 = _mean(yes_fwd[y1])
    acc_y2 = _mean(~yes_fwd[y2])
    y_bin = y1[match].numpy().astype(int)
    return {
        "q_n_y1": float(n1),
        "q_n_y2": float(n2),
        "q_true_yes_share": n1 / max(n1 + n2, 1),
        "q_fwd_yes_share": _mean(yes_fwd[match]),
        "q_fwd_raw_acc": _mean(yes_fwd[match] == y1[match]),
        "q_fwd_acc_y1": acc_y1,
        "q_fwd_acc_y2": acc_y2,
        "q_fwd_balanced_acc": (acc_y1 + acc_y2) / 2,
        "q_fwd_auc": _safe_auc(y_bin, pq_fwd[match].numpy()),
        "q_fwd_best_thr_acc": _best_threshold_acc(y_bin, pq_fwd[match].numpy()),
        "q_rev_yes_share_on_y1": _mean(pq_rev[y1] > 0.5),
        "p_rev_auc_y1_vs_y2": _safe_auc(y_bin, pp_rev[match].numpy()),
    }


def p_diagnostics(
    y_true: Tensor, pp_fwd: Tensor, pp_rev: Tensor, pq_fwd: Tensor
) -> dict[str, float]:
    y0, y1, y2 = y_true == 0, y_true == 1, y_true == 2
    pf, pr, qy = pp_fwd > 0.5, pp_rev > 0.5, pq_fwd > 0.5
    out: dict[str, float] = {}
    for name, mask in (("y0", y0), ("y1", y1), ("y2", y2)):
        for left in (0, 1):
            for right in (0, 1):
                cell = (pf == bool(left)) & (pr == bool(right))
                out[f"joint_{name}_pf{left}_pr{right}"] = _mean(cell[mask])
    missed_y1 = y1 & ~pf
    out["p_rev_yes_share_on_p_missed_y1"] = _mean(pr[missed_y1])
    out["q_yes_share_on_p_missed_y1"] = _mean(qy[missed_y1])
    out["q_yes_share_on_y0"] = _mean(qy[y0])
    out["p_fwd_auc_y1_vs_y0"] = _safe_auc(
        y1[y0 | y1].numpy().astype(int), pp_fwd[y0 | y1].numpy()
    )
    out["p_fwd_auc_y2_vs_y0"] = _safe_auc(
        y2[y0 | y2].numpy().astype(int), pp_fwd[y0 | y2].numpy()
    )
    return out


def direction_diagnostics(y_true: Tensor, p_dir: Tensor) -> dict[str, float]:
    y1, y2 = y_true == 1, y_true == 2
    match = y1 | y2
    y_bin = y2[match].numpy().astype(int)
    return {
        "p_dir_auc_y2_vs_y1": _safe_auc(y_bin, p_dir[match].numpy()),
        "p_dir_median_y1": float(p_dir[y1].median()) if y1.any() else _NAN,
        "p_dir_median_y2": float(p_dir[y2].median()) if y2.any() else _NAN,
    }


def _or_diagnostics(
    y_true: Tensor, pp_fwd: Tensor, pp_rev: Tensor, pq_fwd: Tensor
) -> dict[str, float]:
    y1, y0 = y_true == 1, y_true == 0
    p_or = (pp_fwd > 0.5) | (pp_rev > 0.5)
    q_yes = pq_fwd > 0.5
    y_or = torch.full_like(y_true, 2)
    y_or[~p_or] = 0
    y_or[p_or & q_yes] = 1
    return {
        "or_p_recall_y1": _mean(p_or[y1]),
        "or_p_false_pos_rate": _mean(p_or[y0]),
        "or_bit_q_acc": _mean(((y_true == 1) == (y_or == 1))[y_true > 0]),
        "or_mcc_fwd": float(metrics.matthews_corrcoef(y_true.numpy(), y_or.numpy())),
    }


def _bit_metrics(outputs: BatchOutputs) -> dict[str, float]:
    y_true, y_pred = outputs.y_true, outputs.y_pred
    p_ok = (y_true > 0) & (y_pred > 0)
    return {
        "bit_p_acc": _mean((y_true > 0) == (y_pred > 0)),
        "bit_q_acc": _mean(((y_true == 1) == (y_pred == 1))[y_true > 0]),
        "bit_q_acc_given_p": _mean(((y_true == 1) == (y_pred == 1))[p_ok]),
        "p_recall_y1": _mean(y_pred[y_true == 1] > 0),
        "p_recall_y2": _mean(y_pred[y_true == 2] > 0),
        "p_false_pos_rate": _mean(y_pred[y_true == 0] > 0),
    }


def _q_given_p_metrics(outputs: BatchOutputs) -> dict[str, float]:
    accepted = (outputs.y_pred > 0) & (outputs.y_true > 0)
    base = _mean(outputs.y_true[accepted] == 1)
    return {
        "q_given_p_majority_baseline": max(base, 1 - base),
        "q_fwd_auc_given_p": _safe_auc(
            (outputs.y_true[accepted] == 1).numpy().astype(int),
            outputs.pq_fwd[accepted].numpy(),
        ),
    }


def all_diagnostics(outputs: BatchOutputs) -> dict[str, float]:
    diag = q_diagnostics(outputs.pq_fwd, outputs.pq_rev, outputs.pp_rev, outputs.y_true)
    diag.update(
        p_diagnostics(outputs.y_true, outputs.pp_fwd, outputs.pp_rev, outputs.pq_fwd)
    )
    diag.update(direction_diagnostics(outputs.y_true, outputs.p_dir))
    diag.update(_bit_metrics(outputs))
    diag.update(_q_given_p_metrics(outputs))
    diag.update(
        _or_diagnostics(outputs.y_true, outputs.pp_fwd, outputs.pp_rev, outputs.pq_fwd)
    )
    return diag


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
        self._best_mean = 0.0

    def _interpret_result(
        self,
        model: MultiClassModule,
        batch_fwd: dict[str, torch.Tensor],
        batch_rev: dict[str, torch.Tensor],
        y_true: Tensor,
    ) -> BatchOutputs:
        logits = model(**batch_fwd)
        logits_rev = model(**batch_rev)
        return BatchOutputs(
            y_pred=decode_logits(logits),
            y_pred_rev=decode_logits(logits_rev),
            pq_fwd=_bit_prob(logits, 1),
            pq_rev=_bit_prob(logits_rev, 1),
            pp_fwd=_bit_prob(logits, 0),
            pp_rev=_bit_prob(logits_rev, 0),
            p_dir=_bit_margin(logits, 0) - _bit_margin(logits_rev, 0),
            y_true=y_true,
        )

    @torch.no_grad()
    def _collect(
        self,
        model: MultiClassModule,
        data_loader: DataLoader[AsymmetricMultiClassDataset],
    ) -> BatchOutputs:
        return BatchOutputs.concat(
            [
                self._interpret_result(model, batch_fwd, batch_rev, y_true)
                for batch_fwd, batch_rev, y_true in data_loader
            ]
        )

    @torch.no_grad()
    def _run_model(
        self,
        model: MultiClassModule,
        data: DataLoader[AsymmetricMultiClassDataset],
        metrics: dict | None = None,
    ) -> tuple[bool, dict]:
        outputs = self._collect(model, data)
        if metrics is not None and self._is_evaluating(metrics):
            return self._test_result(outputs, metrics)
        return self._dev_result(outputs)

    def _dev_result(self, outputs: BatchOutputs) -> tuple[bool, dict]:
        mcc_fwd, mcc_rev, mcc_mean = self._mcc(outputs)
        success = mcc_mean > self._best_mean
        if success:
            self._best_mean = mcc_mean
        return success, self._prefixed("dev", outputs, mcc_fwd, mcc_rev, mcc_mean)

    def _test_result(
        self, outputs: BatchOutputs, best_config: dict
    ) -> tuple[bool, dict]:
        mcc_fwd, mcc_rev, mcc_mean = self._mcc(outputs)
        best_config.update(self._prefixed("test", outputs, mcc_fwd, mcc_rev, mcc_mean))
        return True, best_config

    @staticmethod
    def _mcc(outputs: BatchOutputs) -> tuple[float, float, float]:
        y_true = outputs.y_true
        y_true_rev = y_true.clone()
        y_true_rev[y_true == 2] = 0
        mcc_fwd = metrics.matthews_corrcoef(y_true.numpy(), outputs.y_pred.numpy())
        mcc_rev = metrics.matthews_corrcoef(
            y_true_rev.numpy(), outputs.y_pred_rev.numpy()
        )
        return mcc_fwd, mcc_rev, _harmonic_mean(mcc_fwd, mcc_rev)

    def _prefixed(
        self,
        prefix: str,
        outputs: BatchOutputs,
        mcc_fwd: float,
        mcc_rev: float,
        mcc_mean: float,
    ) -> dict[str, float]:
        diag = all_diagnostics(outputs)
        return {
            f"{prefix}_mcc_fwd": mcc_fwd,
            f"{prefix}_mcc_rev": mcc_rev,
            f"{prefix}_mcc_mean": mcc_mean,
            **{f"{prefix}_{key}": value for key, value in diag.items()},
        }
