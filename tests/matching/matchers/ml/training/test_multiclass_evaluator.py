import pytest
import torch
from tensorboard.backend.event_processing import event_accumulator

from matchescu.matching.matchers.ml.multiclass.training._evaluator import (
    BatchOutputs,
    TrainingEvaluator,
    _harmonic_mean,
    all_diagnostics,
    direction_diagnostics,
)


class _StubModel:
    def __init__(self):
        self.training = True
        self._logits = {
            0: torch.tensor([[4.0, -4.0, -4.0, 4.0]]),
            1: torch.tensor([[-4.0, 4.0, -4.0, 4.0]]),
            2: torch.tensor([[-4.0, 4.0, 4.0, -4.0]]),
            10: torch.tensor([[4.0, -4.0, -4.0, 4.0]]),
            11: torch.tensor([[-4.0, 4.0, -4.0, 4.0]]),
            12: torch.tensor([[4.0, -4.0, 4.0, -4.0]]),
        }

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        return self._logits[int(input_ids[0])]

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


def _batch(label: int):
    fwd = {"input_ids": torch.tensor([label])}
    rev = {"input_ids": torch.tensor([label + 10])}
    return fwd, rev, torch.tensor([label])


@pytest.fixture
def loader():
    return [_batch(0), _batch(1), _batch(2), _batch(1)]


@pytest.fixture
def evaluator(tmp_path, loader):
    with TrainingEvaluator("task", loader, loader, tmp_path) as result:
        yield result


def test_batch_outputs_concat():
    first = BatchOutputs(*(torch.tensor([1]) for _ in range(8)))
    second = BatchOutputs(*(torch.tensor([2]) for _ in range(8)))

    combined = BatchOutputs.concat([first, second])

    assert combined.y_true.tolist() == [1, 2]


def test_direction_diagnostics_ranks_one_way_pairs():
    y_true = torch.tensor([1, 1, 1, 2, 2, 2])
    p_dir = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0, 1.1])

    diag = direction_diagnostics(y_true, p_dir)

    assert diag["p_dir_auc_y2_vs_y1"] == pytest.approx(1.0)
    assert diag["p_dir_median_y1"] == pytest.approx(0.1)
    assert diag["p_dir_median_y2"] == pytest.approx(1.0)


def test_all_diagnostics_contains_q_p_and_direction_metrics(evaluator, loader):
    outputs = evaluator._collect(_StubModel(), loader)

    diag = all_diagnostics(outputs)

    assert diag["q_fwd_auc_given_p"] == pytest.approx(1.0)
    assert diag["p_dir_auc_y2_vs_y1"] == pytest.approx(1.0)
    assert diag["bit_q_acc_given_p"] == pytest.approx(1.0)
    assert diag["p_recall_y1"] == pytest.approx(1.0)


def test_harmonic_mean_returns_zero_for_non_positive_values():
    assert _harmonic_mean(0.0, 1.0) == 0.0
    assert _harmonic_mean(0.5, -0.5) == 0.0
    assert _harmonic_mean(0.5, 0.5) == pytest.approx(0.5)


def test_run_model_returns_dev_and_test_prefixes(evaluator, loader):
    success, dev = evaluator._run_model(_StubModel(), loader)

    assert success is True
    assert "dev_mcc_mean" in dev
    assert "dev_q_fwd_auc_given_p" in dev

    test_config = {"is_evaluating": True}
    ok, test = evaluator._run_model(_StubModel(), loader, test_config)

    assert ok is True
    assert "test_mcc_mean" in test
    assert "test_q_fwd_auc_given_p" in test


def test_call_logs_diagnostics_to_tensorboard(evaluator, loader, tmp_path):
    evaluator(_StubModel(), {}, 1)
    evaluator.summary_writer.flush()

    metric_dir = tmp_path / "task_dev_q_fwd_auc_given_p"
    accumulator = event_accumulator.EventAccumulator(str(metric_dir))
    accumulator.Reload()

    assert (tmp_path / "task_dev_p_dir_auc_y2_vs_y1").is_dir()
    assert (tmp_path / "task_dev_bit_q_acc_given_p").is_dir()
    assert (tmp_path / "task_test_q_fwd_auc_given_p").is_dir()
    assert "task" in accumulator.Tags()["scalars"]
    assert accumulator.Scalars("task")[-1].value == pytest.approx(1.0)
