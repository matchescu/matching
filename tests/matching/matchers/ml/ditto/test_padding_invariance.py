from types import SimpleNamespace

import numpy as np
import torch
from transformers import BertConfig, BertModel

from matchescu.matching.matchers.ml.ditto._module import DittoModel


def _tiny_model() -> DittoModel:
    config = BertConfig(
        vocab_size=100,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        pad_token_id=0,
    )
    model = object.__new__(DittoModel)
    torch.nn.Module.__init__(model)
    model._bert = BertModel(config)
    model._classifier = torch.nn.Linear(config.hidden_size, 1)
    model._alpha_aug = 0.8
    return model


def test_attention_mask_marks_padding():
    model = object.__new__(DittoModel)
    model._bert = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))
    x = torch.tensor([[101, 200, 102, 0, 0]])
    assert model._attention_mask(x).tolist() == [[1, 1, 1, 0, 0]]


def test_padding_does_not_change_scores():
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()

    x = torch.tensor([[1, 2, 3, 4, 5]])
    padded = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [1, 2, 3, 0, 0, 0, 0, 0],
        ]
    )

    with torch.no_grad():
        single = model(x)
        batch = model(padded)

    assert torch.allclose(single[0], batch[0], atol=1e-5)


def test_mixda_attention_mask_covers_both_segments():
    captured = {}

    class _RecordingBert:
        config = SimpleNamespace(pad_token_id=0)

        def __call__(self, x, attention_mask=None):
            captured["mask"] = attention_mask
            return (torch.zeros(x.shape[0], x.shape[1], 8),)

    model = object.__new__(DittoModel)
    torch.nn.Module.__init__(model)
    model._bert = _RecordingBert()
    model._classifier = torch.nn.Linear(8, 1)
    model._alpha_aug = 0.8
    np.random.seed(0)

    x1 = torch.tensor([[1, 2, 3, 0, 0]])
    x2 = torch.tensor([[4, 5, 0, 0, 0]])
    model._bert_encode(x1, x2)

    assert captured["mask"].tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ]
