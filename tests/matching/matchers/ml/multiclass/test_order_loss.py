import pytest
import torch

from matchescu.matching.matchers.ml.multiclass import _loss


@pytest.fixture
def embeddings():
    a = torch.tensor([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True)
    b = torch.tensor([[0.0, 1.0], [1.0, 2.0], [0.0, 0.0]], requires_grad=True)
    return a, b


def test_order_energy_is_squared_positive_coordinate_difference(embeddings):
    a, b = embeddings
    torch.testing.assert_close(_loss.order_energy(a, b), torch.tensor([1.0, 5.0, 0.0]))
    torch.testing.assert_close(_loss.order_energy(b, a), torch.tensor([4.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    "labels,expected", [([2, 2, 2], 3.0), ([0, 0, 0], 2.0), ([2, 0, 1], 1.5)]
)
def test_order_loss_uses_class_conditional_terms(embeddings, labels, expected):
    result = _loss.order_loss(*embeddings, torch.tensor(labels), margin=2.0)
    assert result.item() == pytest.approx(expected)


@pytest.mark.parametrize("size", [0, 3])
def test_order_loss_returns_tensor_zero_without_constrained_labels(embeddings, size):
    a, b = embeddings
    result = _loss.order_loss(a[:size].double(), b[:size].double(), torch.ones(size))
    torch.testing.assert_close(result, a.double().new_zeros(()))


def test_order_loss_averages_present_classes_not_samples(embeddings):
    a, b = embeddings
    indexes = torch.tensor([0, 0, 0, 1])
    result = _loss.order_loss(
        a[indexes], b[indexes], torch.tensor([2, 2, 2, 0]), margin=2.0
    )
    assert result.item() == pytest.approx(1.5)


def test_order_loss_leaves_class_one_gradients_unconstrained(embeddings):
    a, b = embeddings
    _loss.order_loss(a, b, torch.tensor([2, 0, 1]), margin=2.0).backward()
    torch.testing.assert_close(a.grad[2], torch.zeros(2))
    torch.testing.assert_close(b.grad[2], torch.zeros(2))


@pytest.mark.parametrize("label,gradient", [(2, [-1.0, -2.0]), (0, [-1.0, 2.0])])
def test_order_loss_backpropagates_exact_gradients(label, gradient):
    a = torch.tensor([[0.5, 0.0]], requires_grad=True)
    b = torch.tensor([[0.0, 1.0]], requires_grad=True)
    _loss.order_loss(a, b, torch.tensor([label]), margin=2.0).backward()
    torch.testing.assert_close(a.grad, torch.tensor([gradient]))
    torch.testing.assert_close(b.grad, -a.grad)


def test_order_loss_gradient_step_improves_class_two_direction():
    a = torch.tensor([[0.5, 0.0]], requires_grad=True)
    b = torch.tensor([[0.0, 1.0]], requires_grad=True)
    before = _loss.order_loss(a, b, torch.tensor([2]), margin=2.0)
    before.backward()
    updated_a, updated_b = a - 0.1 * a.grad, b - 0.1 * b.grad
    after = _loss.order_loss(updated_a, updated_b, torch.tensor([2]), margin=2.0)
    assert after < before
    assert _loss.order_energy(updated_a, updated_b) < _loss.order_energy(a, b)
    assert _loss.order_energy(updated_b, updated_a) > _loss.order_energy(b, a)
