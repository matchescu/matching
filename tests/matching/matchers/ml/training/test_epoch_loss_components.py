from unittest.mock import Mock

import pytest
import torch

from matchescu.matching.matchers.ml.training import BaseTrainer


@pytest.fixture
def epoch_run(make_trainer, monkeypatch):
    trainer = make_trainer()
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    monkeypatch.setattr(trainer, "_create_loss", lambda _: torch.nn.MSELoss())
    monkeypatch.setattr(
        trainer,
        "_forward_pass",
        lambda model, batch, device: (model(batch), torch.zeros_like(batch)),
    )
    return trainer, model, torch.optim.SGD(model.parameters(), lr=0.1)


def test_epoch_backpropagates_total_and_averages_raw_components(epoch_run, monkeypatch):
    trainer, model, optimizer = epoch_run

    def loss_components(epoch, loss_fn, tensors):
        logits, _ = tensors
        raw = logits.mean()
        return {
            "total": 2 * raw,
            "loss_fwd": raw,
            "loss_rev": raw * 0,
            "loss_dir": raw + 1,
        }

    monkeypatch.setattr(trainer, "_compute_loss", loss_components)
    result = trainer._train_one_epoch(
        1, torch.device("cpu"), model, [torch.ones(1, 1)] * 2, optimizer, Mock()
    )
    assert result == pytest.approx(
        {"average_loss": 1.8, "loss_fwd": 0.9, "loss_rev": 0.0, "loss_dir": 1.9}
    )
    torch.testing.assert_close(model.weight, torch.tensor([[0.6]]))


def test_epoch_preserves_scalar_loss_interface(epoch_run, monkeypatch):
    trainer, model, optimizer = epoch_run
    monkeypatch.setattr(
        trainer, "_compute_loss", BaseTrainer._compute_loss.__get__(trainer)
    )
    result = trainer._train_one_epoch(
        1, torch.device("cpu"), model, [torch.ones(1, 1)], optimizer, Mock()
    )
    assert result == {"average_loss": 1.0}


def test_epoch_returns_zero_for_empty_loader(epoch_run):
    trainer, model, optimizer = epoch_run
    result = trainer._train_one_epoch(
        1, torch.device("cpu"), model, [], optimizer, Mock()
    )
    assert result == {"average_loss": 0.0}
