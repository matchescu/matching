import pytest

from matchescu.matching.similarity._numeric import (
    BoundedNumericDifferenceSimilarity,
    BucketedNorm,
)


@pytest.fixture
def sim(request):
    max_diff = 1.0
    if hasattr(request, "param") and isinstance(max_diff, (int, float)):
        max_diff = request.param
    return BoundedNumericDifferenceSimilarity(max_diff)


@pytest.mark.parametrize(
    "a,b,sim,expected",
    [(0, 0.5, 1, 0.5), (0.5, 0, 1, 0.5), (0, 1, 1, 0), (0, 0, 1, 1), (0, 1.01, 1, 0)],
    indirect=["sim"],
)
def test_numeric_diff_similarity(a, b, sim, expected):
    assert sim(a, b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (5.0, 5.0, 1.0),
        (5.0, 5.05, 0.9),
        (5.0, 5.1, 0.9),
        (5.0, 4.51, 0.6),
        (5.0, 5.51, 0.0),
    ],
)
def test_bucketednorm_mapping_upper_bounds_inclusive(a, b, expected):
    sim = BucketedNorm({0.0: 1.0, 0.1: 0.9, 0.5: 0.6})
    assert sim(a, b) == expected


@pytest.mark.parametrize("catch_all", [0.0, -0.5])
def test_bucketednorm_mapping_catch_all(catch_all):
    sim = BucketedNorm({0.0: 1.0}, catch_all=catch_all)
    assert sim(0.0, 0.01) == catch_all


def test_bucketednorm_iterable_identity_values():
    sim = BucketedNorm([0.0, 5.0, 10.0], catch_all=-1.0)

    assert sim(10, 10.0) == 0.0
    assert sim(10, 10.1) == 5.0
    assert sim(10, 15.1) == 10.0
    assert sim(10, 20.01) == -1.0


def test_bucketednorm_handles_missing_values():
    sim = BucketedNorm(
        [0.0, 0.2, 1.0], catch_all=1.0, missing_both=-1.0, missing_either=-0.5
    )

    assert sim(None, None) == -1.0
    assert sim(1.23, None) == -0.5
    assert sim(None, 4.56) == -0.5


def test_bucketednorm_handles_unsupported_data_types():
    sim = BucketedNorm(
        [0.0, 0.2, 1.0], catch_all=1.0, missing_both=-1.0, missing_either=-0.5
    )

    assert sim("foo", "bar") == -0.5
    assert sim("foo", 3.14) == -0.5


def test_bucketednorm_handles_unsorted_mapping():
    # unordered mapping; duplicates collapse by dict semantics (last wins)
    sim = BucketedNorm({0.5: 0.6, 0.0: 1.0, 0.1: 0.9}, catch_all=0.0)

    assert sim(1.0, 1.0) == 1.0
    assert sim(1.0, 1.08) == 0.9
    assert sim(1.0, 1.3) == 0.6
    assert sim(1.0, 2.0) == 0.0


@pytest.mark.parametrize(
    "obj,expected",
    [
        ([], "'buckets' is empty"),
        ({}, "'buckets' is empty"),
        (0, "unsupported 'buckets' type: 'int'"),
    ],
)
def test_bucketednorm_raises_assertion_error(obj, expected):
    with pytest.raises(AssertionError) as err_proxy:
        BucketedNorm(obj)

    assert str(err_proxy.value) == expected
