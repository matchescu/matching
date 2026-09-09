from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams
from matchescu.matching.matchers.ml.multiclass.training import MultiClassTrainer

from ._constants import BATCH, HIDDEN, SEQ


@pytest.fixture
def fake_bert():
    bert = MagicMock(spec=nn.Module)
    bert.config = SimpleNamespace(hidden_size=HIDDEN)
    bert.dtype = torch.float32
    return bert


@pytest.fixture(autouse=True)
def patch_auto_model(monkeypatch, fake_bert):
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.multiclass._module.AutoModel.from_pretrained",
        lambda *args, **kwargs: fake_bert,
    )


@pytest.fixture
def make_params():
    def _make(**overrides):
        return MultiClassTrainingParams(**overrides)

    return _make


@pytest.fixture
def synthetic_hidden():
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    mask_a = torch.zeros(BATCH, SEQ)
    mask_a[:, : SEQ // 2] = 1.0
    mask_b = torch.zeros(BATCH, SEQ)
    mask_b[:, SEQ // 2 :] = 1.0
    return hidden, mask_a.unsqueeze(-1), mask_b.unsqueeze(-1)


@pytest.fixture
def col_positions():
    return torch.tensor([[0, SEQ // 2, -1, -1]] * BATCH)


@pytest.fixture
def mock_data_loader():
    def _make(label_counts):
        loader = MagicMock()
        loader.dataset.label_counts = torch.tensor(label_counts)
        return loader

    return _make


@pytest.fixture
def make_trainer():
    def _make(**overrides):
        return MultiClassTrainer(
            "test", MultiClassTrainingParams(**overrides), model_dir="/tmp"
        )

    return _make


@pytest.fixture
def multiclass_trainer(make_trainer):
    return make_trainer()
