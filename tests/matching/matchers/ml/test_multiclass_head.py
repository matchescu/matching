import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import (
    ArchitectureType,
    HeadType,
)

from ._constants import BATCH, HIDDEN


@pytest.mark.parametrize(
    "head_type,expected_size",
    [
        (HeadType.NONE, 2 * HIDDEN),
        (HeadType.ABS, 3 * HIDDEN),
        ("asymmetric", 3 * HIDDEN + 256 + 1),
    ],
)
def test_classifier_input_size_when_head_type_is(make_params, head_type, expected_size):
    module = MultiClassModule(
        make_params(head_type=head_type, architecture=ArchitectureType.BERT)
    )
    assert module.classifier_input_size == expected_size


def test_apply_head_when_head_type_none_returns_two_chunk_width(make_params):
    module = MultiClassModule(
        make_params(head_type=HeadType.NONE, architecture=ArchitectureType.BERT)
    )
    enc_a = torch.randn(BATCH, HIDDEN)
    enc_b = torch.randn(BATCH, HIDDEN)
    result = module._apply_head(enc_a, enc_b)
    assert result.shape == (BATCH, 2 * HIDDEN)


def test_apply_head_when_head_type_abs_returns_three_chunk_width(make_params):
    module = MultiClassModule(
        make_params(head_type=HeadType.ABS, architecture=ArchitectureType.BERT)
    )
    enc_a = torch.randn(BATCH, HIDDEN)
    enc_b = torch.randn(BATCH, HIDDEN)
    result = module._apply_head(enc_a, enc_b)
    assert result.shape == (BATCH, 3 * HIDDEN)


def test_apply_head_when_head_type_abs_diff_chunk_is_symmetric(make_params):
    module = MultiClassModule(
        make_params(head_type=HeadType.ABS, architecture=ArchitectureType.BERT)
    )
    enc_a = torch.randn(BATCH, HIDDEN)
    enc_b = torch.randn(BATCH, HIDDEN)
    result_ab = module._apply_head(enc_a, enc_b)
    result_ba = module._apply_head(enc_b, enc_a)
    assert torch.allclose(result_ab[:, 2 * HIDDEN :], result_ba[:, 2 * HIDDEN :])


def test_apply_head_when_head_type_asymmetric_returns_default_width(make_params):
    module = MultiClassModule(
        make_params(head_type="asymmetric", architecture=ArchitectureType.BERT)
    )
    enc_a = torch.randn(BATCH, HIDDEN)
    enc_b = torch.randn(BATCH, HIDDEN)
    result = module._apply_head(enc_a, enc_b)
    assert result.shape == (BATCH, 3 * HIDDEN + 256 + 1)


def test_apply_head_when_head_type_asymmetric_order_delta_changes_sign(make_params):
    module = MultiClassModule(
        make_params(head_type="asymmetric", architecture=ArchitectureType.BERT)
    )
    enc_a = torch.randn(BATCH, HIDDEN)
    enc_b = torch.randn(BATCH, HIDDEN)
    result_ab = module._apply_head(enc_a, enc_b)
    result_ba = module._apply_head(enc_b, enc_a)
    torch.testing.assert_close(result_ab[:, -1], -result_ba[:, -1])
