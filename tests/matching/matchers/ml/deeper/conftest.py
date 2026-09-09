from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from matchescu.matching.matchers.ml.deeper._module import DeepERModule
from matchescu.matching.matchers.ml.deeper._params import DeepERParams

from .._constants import BATCH, HIDDEN, SEQ


def _make_fake_bert():
    """A MagicMock BERT exposing the surface DeepERModule touches.

    DeepERModule.__freeze_bert iterates ``embeddings.parameters()``,
    ``encoder.layer`` and ``pooler.parameters()``; __encode_all_attrs calls
    ``bert(input_ids=..., attention_mask=...).last_hidden_state``. A plain
    ``MagicMock(spec=nn.Module)`` has no ``embeddings``/``encoder``/``pooler``
    attributes, so we build a bare MagicMock and wire those up explicitly.
    """
    bert = MagicMock()
    bert.config = SimpleNamespace(hidden_size=HIDDEN)
    bert.dtype = torch.float32
    bert.embeddings = MagicMock()
    bert.embeddings.parameters = lambda: iter(())
    bert.encoder = MagicMock()
    bert.encoder.layer = MagicMock()
    bert.encoder.layer.__iter__ = lambda self: iter(())
    bert.pooler = MagicMock()
    bert.pooler.parameters = lambda: iter(())

    def _call(*args, **kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        out = MagicMock()
        out.last_hidden_state = ids.float().unsqueeze(-1).repeat(1, 1, HIDDEN)
        return out

    bert.side_effect = _call
    return bert


@pytest.fixture
def deeper_fake_bert():
    return _make_fake_bert()


@pytest.fixture(autouse=True)
def patch_bert_model(monkeypatch, deeper_fake_bert):
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.deeper._module.BertModel.from_pretrained",
        lambda *args, **kwargs: deeper_fake_bert,
    )


@pytest.fixture
def make_deeper_params():
    def _make(**overrides):
        defaults = {
            "num_attributes": 2,
            "lstm_hidden_size": HIDDEN,
            "similarity_hidden_size": 8,
            "output_size": 2,
        }
        defaults.update(overrides)
        return DeepERParams(**defaults)

    return _make


@pytest.fixture
def deeper_module(make_deeper_params):
    return DeepERModule(make_deeper_params())


@pytest.fixture
def make_attr():
    def _make(batch=BATCH, seq=SEQ):
        return {
            "input_ids": torch.randint(0, 100, (batch, seq)),
            "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        }

    return _make


@pytest.fixture
def deeper_batch(make_attr, make_deeper_params):
    n_attrs = make_deeper_params().num_attributes
    left = [make_attr() for _ in range(n_attrs)]
    right = [make_attr() for _ in range(n_attrs)]
    return left, right


class _RowSignalLSTM(nn.Module):
    """Mock LSTM that makes the packing-reorder bug observable.

    A real LSTM returns ``h_n`` in the *sorted* order imposed by
    ``pack_padded_sequence(enforce_sorted=False)``. This mock returns a hidden
    state whose value is the *original* row index (broadcast across the hidden
    dim), placed in that sorted order. After a correct ``__compose_attr``
    restores the original order, every output row carries its own original
    index. The pre-fix code returned the rows still in sorted order, so the
    signals came back permuted.
    """

    def forward(self, packed, state=None):
        sorted_indices = packed.sorted_indices
        signal = sorted_indices.float().unsqueeze(-1).expand(-1, HIDDEN)
        return None, (signal.unsqueeze(0), signal.unsqueeze(0))


@pytest.fixture
def row_signal_lstm():
    return _RowSignalLSTM()
