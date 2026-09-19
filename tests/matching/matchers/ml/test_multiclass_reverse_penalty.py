from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from matchescu.matching.matchers.ml.multiclass._loss import FocalLoss
from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams
from matchescu.matching.matchers.ml.multiclass._types import LossType


@pytest.fixture
def loss_tensors():
    fwd = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]], requires_grad=True
    )
    rev = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], requires_grad=True
    )
    return (
        fwd,
        rev,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 0]),
        torch.zeros(3, 2),
        torch.ones(3, 2),
    )


def test_directional_params_replace_reverse_penalty():
    params = MultiClassTrainingParams()
    assert (params.focal_gamma, params.dir_margin_weight, params.dir_margin) == (
        0.0,
        0.0,
        2.0,
    )
    assert "reverse_penalty_weight" not in type(params).model_fields
    assert "reversePenaltyWeight" not in params.model_dump()


@pytest.mark.parametrize("gamma", [0.0, 2.0])
@pytest.mark.parametrize("weight", [0.0, 2.0])
@pytest.mark.parametrize("loss_type", list(LossType))
def test_loss_settings_are_independent(
    make_trainer, mock_data_loader, loss_tensors, gamma, weight, loss_type
):
    trainer = make_trainer(
        loss_type=loss_type, focal_gamma=gamma, dir_margin_weight=weight, dir_margin=3.0
    )
    loss_fn = trainer._create_loss(mock_data_loader([100, 25, 4]))
    expected_gamma = gamma if loss_type == LossType.FOCAL else 0.0
    assert loss_fn.gamma == expected_gamma
    result = trainer._compute_loss(0, loss_fn, loss_tensors)
    fwd, rev, y, y_rev, _, _ = loss_tensors
    torch.testing.assert_close(result["loss_fwd"], loss_fn(fwd, y))
    torch.testing.assert_close(result["loss_rev"], loss_fn(rev, y_rev))
    torch.testing.assert_close(result["loss_dir"], fwd.new_tensor(2.0))
    torch.testing.assert_close(
        result["total"], loss_fn(fwd, y) + loss_fn(rev, y_rev) + weight * 2.0
    )


@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_fixed_baseline_differs_only_by_weighted_focal_reverse(
    make_trainer, mock_data_loader, loss_tensors, gamma
):
    trainer = make_trainer(focal_gamma=gamma)
    loss_fn = trainer._create_loss(mock_data_loader([100, 25, 4]))
    fwd, rev, y, y_rev, _, _ = loss_tensors
    old_fwd = FocalLoss(loss_fn.alpha, gamma=0.0)(fwd, y)
    old_rev = F.cross_entropy(rev, y_rev)
    result = trainer._compute_loss(0, loss_fn, loss_tensors)
    expected_delta = loss_fn(fwd, y) - old_fwd + loss_fn(rev, y_rev) - old_rev
    torch.testing.assert_close(result["total"] - (old_fwd + old_rev), expected_delta)
    assert not torch.isclose(result["loss_rev"], old_rev)


def test_default_loss_matches_old_baseline_for_unit_weights(make_trainer, loss_tensors):
    loss_fn = FocalLoss(torch.ones(3), gamma=0.0)
    fwd, rev, y, y_rev, _, _ = loss_tensors
    result = make_trainer()._compute_loss(0, loss_fn, loss_tensors)
    torch.testing.assert_close(
        result["total"], loss_fn(fwd, y) + F.cross_entropy(rev, y_rev)
    )


def test_compute_loss_reuses_loss_object_for_valid_reverse_targets(
    make_trainer, loss_tensors
):
    fwd, rev, y, _, a, b = loss_tensors
    y_rev = torch.tensor([0, 1, 2])
    loss_fn = Mock(side_effect=lambda x, y: F.cross_entropy(x, y))
    make_trainer()._compute_loss(0, loss_fn, (fwd, rev, y, y_rev, a, b))
    assert loss_fn.call_count == 2
    torch.testing.assert_close(loss_fn.call_args.args[0], rev[:2])
    torch.testing.assert_close(loss_fn.call_args.args[1], y_rev[:2])


def test_forward_pass_relabels_class_two(make_trainer, loss_tensors):
    fwd, rev, y, y_rev, a, b = loss_tensors
    result = make_trainer()._forward_pass(
        Mock(side_effect=[(fwd, a, b), rev]), ({}, {}, y), torch.device("cpu")
    )
    torch.testing.assert_close(result[3], y_rev)
