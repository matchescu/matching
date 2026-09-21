import pytest
import torch

from matchescu.matching.matchers.ml.deeper._params import DeepERParams
from matchescu.matching.matchers.ml.deeper.training import DeepERTrainer
from matchescu.matching.matchers.ml.deepmatcher._params import (
    DeepMatcherModelTrainingParams,
)
from matchescu.matching.matchers.ml.deepmatcher.training import DeepMatcherTrainer
from matchescu.matching.matchers.ml.ditto._params import DittoModelTrainingParams
from matchescu.matching.matchers.ml.ditto.training import DittoTrainer


def _make_trainer(cls, params_cls):
    return cls("test", params_cls(), model_dir="/tmp")


@pytest.fixture
def deeper_trainer():
    return _make_trainer(DeepERTrainer, DeepERParams)


@pytest.fixture
def deepmatcher_trainer():
    return _make_trainer(
        DeepMatcherTrainer,
        lambda: DeepMatcherModelTrainingParams(vocab_size=100, embedding_dim=30),
    )


@pytest.fixture
def ditto_trainer():
    return _make_trainer(DittoTrainer, DittoModelTrainingParams)


def test_deeper_compute_loss_runs_forward_and_backward(deeper_trainer):
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.randint(0, 2, (8,), dtype=torch.long)

    loss = deeper_trainer._compute_loss(0, loss_fn, [logits, targets])

    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_deepmatcher_compute_loss_runs_forward_and_backward(deepmatcher_trainer):
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.randint(0, 2, (8,), dtype=torch.long)

    loss = deepmatcher_trainer._compute_loss(0, loss_fn, [logits, targets])

    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ditto_compute_loss_runs_forward_and_backward(ditto_trainer):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logits = torch.randn(8, requires_grad=True)
    targets = torch.randint(0, 2, (8,), dtype=torch.int64)

    loss = ditto_trainer._compute_loss(0, loss_fn, [logits, targets])

    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
