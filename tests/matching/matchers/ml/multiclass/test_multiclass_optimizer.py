import pytest
import torch
from torch import nn

from matchescu.matching.matchers.ml.multiclass._classifier import ClassificationHead
from matchescu.matching.matchers.ml.multiclass._cross_attention import (
    PerAttributeCrossAttention,
)

from .._constants import HIDDEN


class _StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)


class _StubBert(nn.Module):
    def __init__(self, num_layers: int = 12):
        super().__init__()
        self.embeddings = nn.Linear(HIDDEN, HIDDEN)
        self.encoder_layers = nn.ModuleList(_StubLayer() for _ in range(num_layers))
        first_layer = self.encoder_layers[0]
        for param in first_layer.parameters():
            param.requires_grad = False


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._bert = _StubBert()
        self._classifier = ClassificationHead(
            3 * HIDDEN, HIDDEN, 3, dropout_p=0.1, dtype=torch.float32
        )
        self._cross_attn = PerAttributeCrossAttention(HIDDEN, dropout=0.1)

    @property
    def classifier(self):
        return self._classifier

    @property
    def encoder_layers(self):
        return self._bert.encoder_layers

    @property
    def embeddings_layer(self):
        return self._bert.embeddings

    @property
    def cross_attention(self):
        return self._cross_attn


@pytest.fixture
def stub_model():
    return _StubModel()


def test_optimizer_covers_all_trainable_params(make_trainer, stub_model):
    optimizer = make_trainer()._create_optimizer(stub_model)

    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable = {id(p) for p in stub_model.parameters() if p.requires_grad}

    assert trainable <= optimized
