from unittest.mock import Mock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._loss import FocalLoss
from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams


@pytest.fixture
def order_tensors():
    fwd = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], requires_grad=True)
    rev = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0]], requires_grad=True)
    a = torch.tensor([[0.5, 0.0], [9.0, -9.0]], requires_grad=True)
    b = torch.tensor([[0.0, 1.0], [-9.0, 9.0]], requires_grad=True)
    return fwd, rev, torch.tensor([2, 1]), torch.tensor([0, 1]), a, b


def test_order_params_default_to_disabled():
    params = MultiClassTrainingParams()
    assert (params.order_loss_weight, params.order_margin) == (0.0, 1.0)


def test_order_params_round_trip_camel_case():
    params = MultiClassTrainingParams.model_validate(
        {"orderLossWeight": 2.0, "orderMargin": 3.0}
    )
    assert params.model_dump()["orderLossWeight"] == 2.0
    assert params.model_dump()["orderMargin"] == 3.0


@pytest.mark.parametrize("weight", [0.0, 2.0])
def test_trainer_adds_weighted_order_term_but_reports_raw_loss(
    make_trainer, order_tensors, weight
):
    trainer = make_trainer(
        order_loss_weight=weight, order_margin=2.0, dir_margin_weight=2.0
    )
    result = trainer._compute_loss(0, FocalLoss(gamma=0.0), order_tensors)
    torch.testing.assert_close(result["loss_order"], torch.tensor(2.75))
    expected = (
        result["loss_fwd"]
        + result["loss_rev"]
        + 2.0 * result["loss_dir"]
        + weight * 2.75
    )
    torch.testing.assert_close(result["total"], expected)
    result["total"].backward()
    torch.testing.assert_close(
        order_tensors[4].grad, torch.tensor([[-1.0, -2.0], [0.0, 0.0]]) * weight
    )


def test_trainer_logs_tensor_zero_for_all_class_one(make_trainer, order_tensors):
    fwd, rev, y, y_rev, a, b = order_tensors
    result = make_trainer()._compute_loss(
        0, FocalLoss(gamma=0.0), (fwd, rev, y * 0 + 1, y_rev * 0 + 1, a, b)
    )
    torch.testing.assert_close(result["loss_order"], torch.tensor(0.0))


def test_epoch_logs_raw_order_loss_when_weight_is_zero(
    make_trainer, order_tensors, monkeypatch
):
    trainer = make_trainer()
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(trainer, "_forward_pass", lambda *args: order_tensors)
    monkeypatch.setattr(trainer, "_create_loss", lambda _: FocalLoss(gamma=0.0))
    result = trainer._train_one_epoch(
        1,
        torch.device("cpu"),
        model,
        [None],
        torch.optim.SGD(model.parameters(), lr=0.1),
        Mock(),
    )
    assert result["loss_order"] == pytest.approx(1.75)
