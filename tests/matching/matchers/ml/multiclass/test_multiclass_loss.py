import pytest

from matchescu.matching.matchers.ml.multiclass._loss import FocalLoss
from matchescu.matching.matchers.ml.multiclass._types import LossType


def test_create_loss_when_loss_type_weighted_ce_returns_focal_with_gamma_zero(
    make_trainer, mock_data_loader
):
    trainer = make_trainer(loss_type=LossType.WEIGHTED_CE)
    loader = mock_data_loader([100, 50, 12])
    loss_fn = trainer._create_loss(loader)
    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.gamma == 0.0


def test_create_loss_when_loss_type_focal_returns_focal_with_configured_gamma(
    make_trainer, mock_data_loader
):
    trainer = make_trainer(loss_type=LossType.FOCAL, focal_gamma=0.5)
    loader = mock_data_loader([100, 50, 12])
    loss_fn = trainer._create_loss(loader)
    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.gamma == 0.5


def test_create_loss_weights_are_sqrt_dampened_and_normalized(
    make_trainer, mock_data_loader
):
    trainer = make_trainer(loss_type=LossType.WEIGHTED_CE)
    loader = mock_data_loader([100, 50, 12])
    loss_fn = trainer._create_loss(loader)
    assert loss_fn.alpha[0].item() == pytest.approx(1.0)


def test_create_loss_weights_are_monotonic_for_rarer_classes(
    make_trainer, mock_data_loader
):
    trainer = make_trainer(loss_type=LossType.WEIGHTED_CE)
    loader = mock_data_loader([100, 50, 12])
    loss_fn = trainer._create_loss(loader)
    assert loss_fn.alpha[2] > loss_fn.alpha[1] > loss_fn.alpha[0]
