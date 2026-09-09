from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass.training._evaluator import (
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
        return pred, pred_rev

    monkeypatch.setattr(TrainingEvaluator, "_interpret_result", _interpret)


@pytest.fixture
def evaluator():
    return _make_evaluator()


@pytest.fixture
def patched_evaluator(monkeypatch, evaluator):
    return evaluator, monkeypatch


def _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true):
    _patch_interpret(monkeypatch, y_pred, y_pred_rev)
    evaluator._xv_data = [_fake_batch(y_true)]
    return evaluator._run_model(MagicMock(), evaluator._xv_data, {"average_loss": 1.0})


def test_class2_metrics_forward_fn(patched_evaluator):
    """Class-2 true labels predicted as non-2 count as forward FN."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 1, 0, 1]
    y_pred_rev = [0, 1, 0, 0, 1]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    n = len(y_true)
    assert result["dev_c2_fwd_fn"] == pytest.approx(2 / n)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0 / n)
    assert result["dev_c2_rev_fp"] == pytest.approx(0 / n)


def test_class2_metrics_forward_fp(patched_evaluator):
    """Non-2 true labels predicted as 2 count as forward FP."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [2, 1, 0, 2, 1]
    y_pred_rev = [0, 1, 0, 0, 1]
    y_true = [0, 1, 0, 1, 1]

    _, result = _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    n = len(y_true)
    assert result["dev_c2_fwd_fn"] == pytest.approx(0 / n)
    assert result["dev_c2_fwd_fp"] == pytest.approx(2 / n)
    assert result["dev_c2_rev_fp"] == pytest.approx(0 / n)


def test_class2_metrics_reverse_fp(patched_evaluator):
    """Any class-2 prediction in the reverse direction is a false positive,
    since y_true_rev has all class-2 labels relabeled to 0."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 2, 2, 0, 2]
    y_true = [0, 1, 2, 2, 1]

    _, result = _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    n = len(y_true)
    assert result["dev_c2_fwd_fn"] == pytest.approx(1 / n)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0 / n)
    assert result["dev_c2_rev_fp"] == pytest.approx(3 / n)


def test_class2_metrics_combined(patched_evaluator):
    """All three error types present in a single batch."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [2, 1, 1, 0, 2, 2]
    y_pred_rev = [0, 2, 0, 2, 2, 0]
    y_true = [0, 1, 2, 2, 1, 2]

    _, result = _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    n = len(y_true)
    # y_true==2 at idx 2,3,5; y_pred!=2 at 2,3 -> FN=2
    # y_true!=2 at idx 0,1,4; y_pred==2 at 0,4 -> FP=2
    # y_pred_rev==2 at idx 1,3,4 -> rev_fp=3
    assert result["dev_c2_fwd_fn"] == pytest.approx(2 / n)
    assert result["dev_c2_fwd_fp"] == pytest.approx(2 / n)
    assert result["dev_c2_rev_fp"] == pytest.approx(3 / n)


def test_class2_metrics_zero_errors(patched_evaluator):
    """A perfect forward prediction and no class-2 reverse predictions yield 0."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1, 2]
    y_pred_rev = [0, 1, 0, 0, 1, 0]
    y_true = [0, 1, 2, 0, 1, 2]

    _, result = _run_dev(evaluator, monkeypatch, y_pred, y_pred_rev, y_true)

    assert result["dev_c2_fwd_fn"] == pytest.approx(0.0)
    assert result["dev_c2_fwd_fp"] == pytest.approx(0.0)
    assert result["dev_c2_rev_fp"] == pytest.approx(0.0)


def test_test_branch_includes_class2_metrics(patched_evaluator):
    """The evaluation branch (is_evaluating=True) must populate test_ keys."""
    evaluator, monkeypatch = patched_evaluator
    y_pred = [0, 1, 2, 0, 1]
    y_pred_rev = [0, 1, 0, 0, 2]
    y_true = [0, 1, 2, 0, 1]
    _patch_interpret(monkeypatch, y_pred, y_pred_rev)

    evaluator._test_data = [_fake_batch(y_true)]

    best_config = {"is_evaluating": True, "average_loss": 0.5}
    ok, result = evaluator._run_model(MagicMock(), evaluator._test_data, best_config)

    assert ok is True
    assert "test_c2_fwd_fn" in result
    assert "test_c2_fwd_fp" in result
    assert "test_c2_rev_fp" in result
    n = len(y_true)
    assert result["test_c2_rev_fp"] == pytest.approx(1 / n)
