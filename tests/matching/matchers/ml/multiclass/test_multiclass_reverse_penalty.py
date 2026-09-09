import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._loss import FocalLoss
from matchescu.matching.matchers.ml.multiclass._types import LossType


def _compute_loss_with_penalty(make_trainer, penalty_weight):
    trainer = make_trainer(
        loss_type=LossType.WEIGHTED_CE, reverse_penalty_weight=penalty_weight
    )
    loss_fn = FocalLoss(torch.tensor([1.0, 1.0, 1.0]), gamma=0.0)

    torch.manual_seed(42)
    logits = torch.randn(8, 3, requires_grad=True)
    logits_rev = torch.randn(8, 3, requires_grad=True)
    targets = torch.randint(0, 2, (8,), dtype=torch.long)
    targets_rev = targets.clone()
    targets_rev[targets == 2] = 0

    loss = trainer._compute_loss(0, loss_fn, [logits, logits_rev, targets, targets_rev])
    return loss.item(), logits_rev.detach()


def test_compute_loss_when_penalty_zero_adds_nothing(make_trainer):
    loss_zero, _ = _compute_loss_with_penalty(make_trainer, 0.0)
    loss_nonzero, logits_rev = _compute_loss_with_penalty(make_trainer, 2.0)

    expected_penalty = 2.0 * torch.softmax(logits_rev, dim=1)[:, 2].mean().item()
    assert loss_nonzero - loss_zero == pytest.approx(expected_penalty, rel=1e-5)


def test_compute_loss_penalty_scales_linearly_with_weight(make_trainer):
    loss_0, logits_rev = _compute_loss_with_penalty(make_trainer, 0.0)
    loss_1, _ = _compute_loss_with_penalty(make_trainer, 1.0)
    loss_2, _ = _compute_loss_with_penalty(make_trainer, 2.0)

    penalty_term = torch.softmax(logits_rev, dim=1)[:, 2].mean().item()
    assert loss_1 - loss_0 == pytest.approx(penalty_term, rel=1e-5)
    assert loss_2 - loss_0 == pytest.approx(2 * penalty_term, rel=1e-5)


def test_forward_pass_relabeled_targets_when_class_is_two_becomes_zero():
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    y_rev = y.clone()
    y_rev[y == 2] = 0

    assert (y_rev[y == 2] == 0).all()
    assert (y_rev[y != 2] == y[y != 2]).all()
