import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._cross_attention import (
    PerAttributeCrossAttention,
)

from .._constants import HIDDEN


@pytest.fixture
def attention():
    module = PerAttributeCrossAttention(HIDDEN, dropout=0.0)
    module.eval()
    return module


def test_attention_output_ignores_masked_span_positions(attention):
    torch.manual_seed(0)
    valid = torch.randn(1, 1, HIDDEN)
    garbage = torch.randn(1, 3, HIDDEN)
    hidden_with = torch.cat([valid, garbage], dim=1)
    hidden_without = valid
    mask_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask_a_without = torch.tensor([[1.0]])
    mask_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask_b_without = torch.tensor([[1.0]])
    col_positions = torch.tensor([[0]])
    col_positions_without = torch.tensor([[0]])

    enc_a_with, enc_b_with = attention.forward(
        hidden_with, mask_a, mask_b, col_positions
    )
    enc_a_without, enc_b_without = attention.forward(
        hidden_without, mask_a_without, mask_b_without, col_positions_without
    )

    assert torch.allclose(enc_a_with, enc_a_without)
    assert torch.allclose(enc_b_with, enc_b_without)
