import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._asymmetry import AsymmetricHead
from matchescu.matching.matchers.ml.multiclass._types import HeadType

DEFAULT_INPUT_SIZE = 8
DEFAULT_RANK = 10
BATCH_SIZE = 5


@pytest.fixture
def input_size(request):
    return int(getattr(request, "param", DEFAULT_INPUT_SIZE))


@pytest.fixture
def rank(request):
    return int(getattr(request, "param", DEFAULT_RANK))


@pytest.fixture
def head_type(request):
    return getattr(request, "param", HeadType.NONE)


@pytest.fixture
def enc_a(input_size):
    return torch.randn(BATCH_SIZE, input_size)


@pytest.fixture
def enc_b(input_size):
    return torch.randn(BATCH_SIZE, input_size)


@pytest.fixture
def asymmetric_head(input_size, rank, head_type):
    return AsymmetricHead(input_size, rank, head_type)


@pytest.mark.parametrize(
    "input_size,rank,head_type,expected",
    [
        (DEFAULT_INPUT_SIZE, DEFAULT_RANK, HeadType.NONE, 16),
        (DEFAULT_INPUT_SIZE, DEFAULT_RANK, HeadType.DIFF, 32),
        (DEFAULT_INPUT_SIZE, DEFAULT_RANK, HeadType.BILINEAR, 34),
        (10, DEFAULT_RANK, HeadType.NONE, 20),
        (10, 15, HeadType.BILINEAR, 45),
    ],
    indirect=["input_size", "rank", "head_type"],
)
def test_asymmetric_head_output_size_matches_output_tensor_size(
    asymmetric_head, enc_a, enc_b, expected
):
    actual = asymmetric_head(enc_a, enc_b)

    assert asymmetric_head.output_size == expected
    assert actual.shape == (BATCH_SIZE, asymmetric_head.output_size)
