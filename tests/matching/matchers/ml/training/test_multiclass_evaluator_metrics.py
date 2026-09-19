from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass.training._evaluator import (
    BestByDevMetric,
    TrainingEvaluator,
)


def _make_evaluator():
    xv = MagicMock()
    test = MagicMock()
    return TrainingEvaluator("test", xv, test, Path("/tmp"))


def _fake_batch(y_true):
    return (
        {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
        {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
        torch.tensor(y_true, dtype=torch.long),
    )


def _patch_interpret(monkeypatch, y_pred, y_pred_rev):
    pred = torch.tensor(y_pred, dtype=torch.long)
    pred_rev = torch.tensor(y_pred_rev, dtype=torch.long)

    def _interpret(self, model, batch_fwd, batch_rev):
        return pred, pred_rev, torch.zeros(len(pred)), torch.zeros(len(pred))

    monkeypatch.setattr(TrainingEvaluator, "_interpret_result", _interpret)


def _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true, *, is_eval=False):
    _patch_interpret(monkeypatch, y_pred, y_pred_rev)
    data = [_fake_batch(y_true)]
    best_config = {"average_loss": 1.0, "is_evaluating": is_eval}
    model = MagicMock(order_margin=1.0)
    model._training_epoch = 1
    return evaluator._run_model(model, data, best_config)


@pytest.fixture
def evaluator():
    return _make_evaluator()


@pytest.fixture
def patched_evaluator(monkeypatch, evaluator):
    return evaluator, monkeypatch


def test_class2_metrics_forward_fn(patched_evaluator):
    """Class-2 true labels predicted as non-2 count as forward FN."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 1, 0, 1]
    y_pred_rev = [0, 1, 0, 0, 1]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    assert result["dev_c2_fwd_fn"] == pytest.approx(1.0)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.0)
    assert result["dev_rev_fpr"] == pytest.approx(0.0)
    assert result["dev_rev_fnr"] == pytest.approx(0.0)


def test_class2_metrics_forward_fp(patched_evaluator):
    """Non-2 true labels predicted as 2 count as forward FP."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [2, 1, 0, 2, 1]
    y_pred_rev = [0, 1, 0, 0, 1]
    y_true = [0, 1, 0, 1, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # no true class 2 in this batch
    assert result["dev_c2_fwd_fn"] != result["dev_c2_fwd_fn"]  # NaN
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.4)
    # y_true==1 at indices 1,3,4; rev_pred==0 only at index 3 -> 1/3
    assert result["dev_rev_fpr"] == pytest.approx(1 / 3)
    assert result["dev_rev_fnr"] == pytest.approx(0.0)


def test_class2_metrics_reverse_fpr(patched_evaluator):
    """Reverse predictions of class 0 on true class 1 are reverse FPR."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 0, 0, 0, 0]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # y==2 at idx 2,3; pred!=2 at idx 3 -> 1/2
    assert result["dev_c2_fwd_fn"] == pytest.approx(0.5)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.0)
    assert result["dev_rev_fpr"] == pytest.approx(1.0)
    assert result["dev_rev_fnr"] == pytest.approx(0.0)


def test_class2_metrics_reverse_fnr(patched_evaluator):
    """Reverse predictions not 0 for true class 0 or 2 are reverse FNR."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [2, 1, 2, 1, 1]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # y==2 at idx 2,3; pred!=2 at idx 3 -> 1/2
    assert result["dev_c2_fwd_fn"] == pytest.approx(0.5)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.0)
    # y==1 at idx 1,4; rev pred is 1 at both, so no reverse FPR
    assert result["dev_rev_fpr"] == pytest.approx(0.0)
    # non-class-1 rev preds at idx 0,2,3 are all wrong
    assert result["dev_rev_fnr"] == pytest.approx(1.0)


def test_class2_metrics_order_acc(patched_evaluator):
    """order_acc counts y==2 with pred_fwd==2 and pred_rev==0."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 2, 1]
    y_pred_rev = [0, 0, 0, 1, 1]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # y==2 at indices 2 and 3; only index 2 has pred==2 and rev==0
    assert result["dev_order_acc"] == pytest.approx(0.5)


def test_collapse_fraction(patched_evaluator):
    """collapse is the fraction where pred_fwd equals pred_rev."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 1, 0, 0, 2]
    y_true = [0, 1, 2, 0, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # equal at indices 0, 1, 3 -> 3/5
    assert result["dev_collapse"] == pytest.approx(0.6)


def test_class2_metrics_combined(patched_evaluator):
    """All error types present in a single batch."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [2, 1, 1, 0, 2, 2]
    y_pred_rev = [0, 0, 0, 0, 0, 0]
    y_true = [0, 1, 2, 2, 1, 2]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    # y_true==2 at idx 2,3,5; y_pred!=2 at 2,3 -> FN=2/3
    assert result["dev_c2_fwd_fn"] == pytest.approx(2 / 3)
    # y_true!=2 at idx 0,1,4; y_pred==2 at 0,4 -> FP=2/3
    assert result["dev_c2_fwd_fp"] == pytest.approx(2 / 3)
    # y_true==1 at idx 1,4; rev pred==0 at both -> FPR=1.0
    assert result["dev_rev_fpr"] == pytest.approx(1.0)


def test_class2_metrics_zero_errors(patched_evaluator):
    """Perfect forward prediction and no class-2 reverse predictions yield 0."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1, 2]
    y_pred_rev = [0, 1, 0, 0, 1, 0]
    y_true = [0, 1, 2, 0, 1, 2]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    assert result["dev_c2_fwd_fn"] == pytest.approx(0.0)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.0)
    assert result["dev_rev_fpr"] == pytest.approx(0.0)
    assert result["dev_rev_fnr"] == pytest.approx(0.0)
    assert result["dev_order_acc"] == pytest.approx(1.0)


def test_test_branch_includes_directional_metrics(patched_evaluator):
    """The evaluation branch (is_evaluating=True) must populate test_ keys."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 0, 0, 0, 0]
    y_true = [0, 1, 2, 0, 1]

    ok, result = _run_model(
        evaluator, monkeypatch, y_pred, y_pred_rev, y_true, is_eval=True
    )

    assert ok is True
    assert "test_c2_fwd_fn" in result
    assert "test_c2_fwd_fp" in result
    assert "test_rev_fpr" in result
    assert "test_rev_fnr" in result
    assert "test_order_acc" in result
    assert "test_collapse" in result
    # y_true==1 at indices 1 and 4; rev_pred==0 at both -> FPR = 1.0
    assert result["test_rev_fpr"] == pytest.approx(1.0)


def test_dev_result_has_mcc(patched_evaluator):
    """dev_mcc is present so the selector can use it."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 1, 0, 0, 1]
    y_true = [0, 1, 2, 0, 1]

    _, result = _run_model(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    assert "dev_mcc" in result
    assert "dev_mcc_rev" in result
    assert "dev_order_c0_support" in result
    assert "dev_order_c1_support" in result
    assert "dev_order_c2_support" in result


def test_best_by_dev_metric_selects_improvements():
    selector = BestByDevMetric("dev_mcc")

    assert selector(1, {"dev_mcc": 0.5}) is True
    assert selector.best_epoch == 1
    assert selector(2, {"dev_mcc": 0.7}) is True
    assert selector.best_epoch == 2
    assert selector(3, {"dev_mcc": 0.65}) is False
    assert selector.best_epoch == 2


def test_best_by_dev_metric_min_delta_ignores_tiny_gains():
    selector = BestByDevMetric("dev_mcc", min_delta=0.01)

    assert selector(1, {"dev_mcc": 0.5}) is True
    assert selector(2, {"dev_mcc": 0.505}) is False
    assert selector(3, {"dev_mcc": 0.52}) is True
