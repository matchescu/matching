import math
from unittest.mock import Mock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass.training._evaluator import (
    TrainingEvaluator,
)


@pytest.fixture
def energy_evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.training._evaluator.SummaryWriter", Mock()
    )
    return TrainingEvaluator("test", [], [], tmp_path)


@pytest.fixture
def split_results():
    labels = torch.tensor([2, 0, 2, 2, 1, 0])
    return (
        labels,
        torch.tensor([0.0, 1.0, 2.0, 4.0, 9.0, 5.0]),
        torch.tensor([1.0, 2.0, 4.0, 9.0, 16.0, 6.0]),
    )


@pytest.fixture
def run_split(energy_evaluator, monkeypatch):
    def run(labels, fwd, rev, sizes=(1, 5), margin=1.0, test=False):
        batches = [(Mock(), Mock(), part) for part in labels.split(sizes)]
        results = [
            (part, part.masked_fill(part == 2, 0), ef, er)
            for part, ef, er in zip(
                labels.split(sizes), fwd.split(sizes), rev.split(sizes)
            )
        ]
        monkeypatch.setattr(
            energy_evaluator, "_interpret_result", Mock(side_effect=results)
        )
        return energy_evaluator._run_model(
            Mock(order_margin=margin),
            batches,
            {"average_loss": 1.0, "is_evaluating": test},
        )[1]

    return run


@pytest.mark.parametrize(
    "label,expected",
    [
        (0, (2, 3.0, 2.0, 4.0, 2.0)),
        (1, (1, 9.0, 0.0, 16.0, 0.0)),
        (2, (3, 2.0, math.sqrt(8 / 3), 14 / 3, math.sqrt(98 / 9))),
    ],
)
def test_diagnostics_use_split_global_population_statistics(
    run_split, split_results, label, expected
):
    result = run_split(*split_results)
    keys = ("support", "fwd_mean", "fwd_std", "rev_mean", "rev_std")
    assert tuple(result[f"dev_order_c{label}_{key}"] for key in keys) == pytest.approx(
        expected
    )


def test_diagnostics_are_independent_of_batch_partition(run_split, split_results):
    first = run_split(*split_results)
    second = run_split(*split_results, sizes=(3, 3))
    assert first == pytest.approx(second)


@pytest.mark.parametrize("margin,expected", [(1.0, 2 / 3), (4.0, 1 / 3), (9.0, 0.0)])
def test_class_two_reverse_fraction_uses_strict_margin(
    run_split, split_results, margin, expected
):
    result = run_split(*split_results, margin=margin)
    assert result["dev_order_c2_rev_above_margin"] == pytest.approx(expected)


@pytest.mark.parametrize("label", [0, 1, 2])
def test_diagnostics_report_nan_when_class_support_is_absent(run_split, label):
    present = (label + 1) % 3
    result = run_split(
        torch.tensor([present]), torch.ones(1), torch.ones(1), sizes=(1,)
    )
    assert result[f"dev_order_c{label}_support"] == 0
    assert all(
        math.isnan(result[f"dev_order_c{label}_{key}"])
        for key in ("fwd_mean", "fwd_std", "rev_mean", "rev_std")
    )
    if label == 2:
        assert math.isnan(result["dev_order_c2_rev_above_margin"])


def test_test_split_receives_energy_diagnostics(run_split, split_results):
    result = run_split(*split_results, test=True)
    assert result["test_order_c2_support"] == 3
    assert result["test_order_c2_rev_above_margin"] == pytest.approx(2 / 3)


def test_interpret_uses_forward_pair_once_for_both_energies(energy_evaluator):
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    a, b = torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    model = Mock(side_effect=[(logits, a, b), logits.flip(-1)])
    pred, pred_rev, fwd, rev = energy_evaluator._interpret_result(
        model, {"input_ids": torch.tensor([1])}, {"input_ids": torch.tensor([2])}
    )
    torch.testing.assert_close(fwd, torch.tensor([1.0]))
    torch.testing.assert_close(rev, torch.tensor([4.0]))
    assert model.call_count == 2
    assert model.call_args_list[0].kwargs["return_embeddings"] is True
    assert "return_embeddings" not in model.call_args_list[1].kwargs
    assert (pred.item(), pred_rev.item()) == (2, 0)


def test_evaluator_logs_energy_metrics_on_epochs_without_improvement(
    energy_evaluator, monkeypatch, split_results
):
    labels, fwd, rev = split_results
    energy_evaluator._xv_data = [(Mock(), Mock(), labels)]
    energy_evaluator._test_data = [(Mock(), Mock(), labels)]
    interpret = Mock(
        return_value=(labels, labels.masked_fill(labels == 2, 0), fwd, rev)
    )
    monkeypatch.setattr(energy_evaluator, "_interpret_result", interpret)
    model = Mock(order_margin=1.0, training=True)
    assert energy_evaluator(model, {"average_loss": 1.0}, 1)[0]
    assert not energy_evaluator(model, {"average_loss": 1.0}, 2)[0]
    assert interpret.call_count == 3
    logged = energy_evaluator.summary_writer.add_scalars.call_args_list
    assert [call.args[2] for call in logged] == [1, 2]
    assert all(call.args[1]["dev_order_c2_support"] == 3 for call in logged)
