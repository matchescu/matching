import pytest

from matchescu.matching.matchers.ml.multiclass._cross_attention import (
    PerAttributeCrossAttention,
    PooledCrossAttention,
)
from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import (
    ArchitectureType,
    HeadType,
)

from ._constants import BATCH, HIDDEN


@pytest.mark.parametrize(
    "architecture,expected_type",
    [
        (ArchitectureType.BERT, type(None)),
        (ArchitectureType.BERT_CROSS_ATTN, PooledCrossAttention),
        (ArchitectureType.BERT_PER_ATTR_CROSS_ATTN, PerAttributeCrossAttention),
    ],
)
def test_cross_attention_when_architecture_is(make_params, architecture, expected_type):
    module = MultiClassModule(
        make_params(head_type=HeadType.NONE, architecture=architecture)
    )
    assert isinstance(module.cross_attention, expected_type)


@pytest.mark.parametrize(
    "architecture",
    [ArchitectureType.BERT, ArchitectureType.BERT_CROSS_ATTN],
)
def test_compute_encodings_without_col_positions_returns_correct_shape(
    make_params, synthetic_hidden, architecture
):
    module = MultiClassModule(
        make_params(head_type=HeadType.NONE, architecture=architecture)
    )
    hidden, mask_a, mask_b = synthetic_hidden
    enc_a, enc_b = module._compute_encodings(hidden, mask_a, mask_b, None)
    assert enc_a.shape == (BATCH, HIDDEN)
    assert enc_b.shape == (BATCH, HIDDEN)


def test_compute_encodings_when_per_attr_without_col_positions_raises_value_error(
    make_params, synthetic_hidden
):
    module = MultiClassModule(
        make_params(
            head_type=HeadType.NONE,
            architecture=ArchitectureType.BERT_PER_ATTR_CROSS_ATTN,
        )
    )
    hidden, mask_a, mask_b = synthetic_hidden
    with pytest.raises(ValueError, match="col_positions required"):
        module._compute_encodings(hidden, mask_a, mask_b, None)


def test_compute_encodings_when_per_attr_with_col_positions_returns_correct_shape(
    make_params, synthetic_hidden, col_positions
):
    module = MultiClassModule(
        make_params(
            head_type=HeadType.NONE,
            architecture=ArchitectureType.BERT_PER_ATTR_CROSS_ATTN,
        )
    )
    hidden, mask_a, mask_b = synthetic_hidden
    enc_a, enc_b = module._compute_encodings(hidden, mask_a, mask_b, col_positions)
    assert enc_a.shape == (BATCH, HIDDEN)
    assert enc_b.shape == (BATCH, HIDDEN)
