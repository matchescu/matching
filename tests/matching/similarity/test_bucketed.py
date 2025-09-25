from typing import Union

import pytest
from unittest.mock import MagicMock, call

from matchescu.matching.similarity._bucketed import BucketedSimilarity
from matchescu.matching.similarity._common import Similarity


@pytest.fixture
def catch_all(request) -> float:
    return request.param if hasattr(request, "param") else -1.0


@pytest.fixture
def missing_both(request) -> float:
    return request.param if hasattr(request, "param") else -1.0


@pytest.fixture
def missing_either(request) -> float:
    return request.param if hasattr(request, "param") else -0.5


@pytest.fixture
def buckets_map() -> dict[float, float]:
    return {0.0: 1.0, 0.1: 0.9, 0.5: 0.6}


@pytest.fixture
def buckets_list() -> list[float]:
    return [0.0, 0.2, 1.0]


@pytest.fixture
def mock_sim(request) -> Union[Similarity[float], MagicMock]:
    return_value = 0.0
    if hasattr(request, "param"):
        return_value = float(request.param)

    m = MagicMock(spec=Similarity[float])
    m.return_value = return_value
    return m


@pytest.fixture
def sut_with_map(
    mock_sim, buckets_map, catch_all, missing_either, missing_both
) -> BucketedSimilarity:
    return BucketedSimilarity(
        mock_sim, buckets_map, catch_all, missing_both, missing_either
    )


@pytest.fixture
def sut_with_list(
    mock_sim, buckets_list, catch_all, missing_either, missing_both
) -> BucketedSimilarity:
    return BucketedSimilarity(
        mock_sim, buckets_list, catch_all, missing_both, missing_either
    )


@pytest.mark.parametrize(
    "mock_sim,expected",
    [(0.0, 1.0), (0.1, 0.9), (0.2, 0.6), (0.3, 0.6), (0.501, -1.0)],
    indirect=["mock_sim"],
)
def test_mapping_rules(sut_with_map, mock_sim, expected):
    a, b = object(), object()
    assert sut_with_map(a, b) == expected
    assert mock_sim.call_args_list == [call(a, b)]


@pytest.mark.parametrize(
    "mock_sim,expected",
    [(0.0, 0.0), (0.1, 0.2), (0.2, 0.2), (0.201, 1.0), (1.0, 1.0), (1.00001, -1.0)],
    indirect=["mock_sim"],
)
def test_iterable_identity_rules(sut_with_list, mock_sim, expected):
    a, b = object(), object()
    assert sut_with_list(a, b) == expected
    assert mock_sim.call_args_list == [call(a, b)]


@pytest.mark.parametrize(
    "mock_sim,expected", [(-0.0001, -0.0001)], indirect=["mock_sim"]
)
def test_negative_passthrough_from_wrapped(
    sut_with_list, sut_with_map, mock_sim, expected
):
    a, b = object(), object()
    assert sut_with_list(a, b) == expected
    assert sut_with_map(a, b) == expected

    assert mock_sim.call_args_list == 2 * [call(a, b)]


@pytest.mark.parametrize(
    "a,b,missing_both,missing_either,expected",
    [
        (None, None, 42, 43, 42),
        (None, object(), 41, 42, 42),
        (object(), None, 41, 42, 42),
    ],
    indirect=["missing_both", "missing_either"],
)
def test_missing_handling_and_wrapped_not_called_on_none(
    sut_with_list, sut_with_map, a, b, mock_sim, expected
):
    assert sut_with_map(a, b) == expected
    assert sut_with_list(a, b) == expected

    assert mock_sim.call_count == 0


@pytest.mark.parametrize(
    "buckets,message",
    [
        ([], "'buckets' is empty"),
        ({}, "'buckets' is empty"),
        (object(), "unsupported 'buckets' type: 'object'"),
        (None, "unsupported 'buckets' type: 'NoneType'"),
    ],
)
def test_invalid_buckets_raise(buckets, mock_sim, message):
    with pytest.raises(AssertionError) as err_proxy:
        BucketedSimilarity(mock_sim, buckets)

    assert str(err_proxy.value) == message
