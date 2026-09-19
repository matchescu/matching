import pytest
import torch

from matchescu.matching.matchers.ml.multiclass import _loss


@pytest.fixture
def directional_logits():
    fwd = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 5.0]])
    rev = torch.tensor([[1.0, 2.0, 2.0], [3.0, 2.0, 8.0], [0.0, 0.0, 1.0]])
    return fwd.requires_grad_(), rev.requires_grad_(), torch.tensor([2, 0, 2])


@pytest.mark.parametrize("margin,expected", [(2.0, 0.5), (0.0, 0.0), (5.0, 2.5)])
def test_directional_loss_uses_class_two_hinge(directional_logits, margin, expected):
    result = _loss.directional_margin_loss(*directional_logits, margin=margin)
    assert result.item() == pytest.approx(expected)


def test_directional_loss_ignores_shared_class_two_shift(directional_logits):
    fwd, rev, y = directional_logits
    shift = torch.tensor([0.0, 0.0, 100.0])
    actual = _loss.directional_margin_loss(fwd + shift, rev + shift, y)
    torch.testing.assert_close(actual, _loss.directional_margin_loss(fwd, rev, y))


def test_directional_loss_returns_scalar_zero_without_support(directional_logits):
    fwd, rev, y = directional_logits
    result = _loss.directional_margin_loss(fwd.double(), rev.double(), y * 0)
    torch.testing.assert_close(result, fwd.double().new_zeros(()))


def test_directional_loss_backpropagates_only_active_hinge(directional_logits):
    fwd, rev, y = directional_logits
    _loss.directional_margin_loss(fwd, rev, y).backward()
    expected = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(fwd.grad, expected)
    torch.testing.assert_close(rev.grad, -expected)
