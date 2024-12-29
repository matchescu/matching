import pytest

from matchescu.matching.entity_reference._comparison import (
    FellegiSunterComparison,
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets import RecordLinkageDataSet


@pytest.fixture
def comparison_config() -> EntityReferenceComparisonConfig:
    return (
        FellegiSunterComparison()
        .jaccard("name", 1, 1)
        .jaccard("description", 2, 2)
        .exact("price", 3, 4)
    )


@pytest.fixture
def attr_comparison_rl(left_source, right_source, true_matches, comparison_config):
    return (
        RecordLinkageDataSet(left_source, right_source, true_matches)
        .attr_compare(comparison_config)
        .cross_sources()
    )


@pytest.fixture
def pm_comparison_rl(
    left_source, right_source, true_matches, comparison_config, request
):
    if not hasattr(request, "param") or not isinstance(request.param, int):
        pytest.fail("pm_comparison depends on the number of possible outcomes")
    return (
        RecordLinkageDataSet(left_source, right_source, true_matches)
        .pattern_encoded(comparison_config, request.param)
        .cross_sources()
    )


def test_target_vector(attr_comparison_rl, left_source, right_source, true_matches):
    expected_size = len(left_source) * len(right_source)

    result = attr_comparison_rl.target_vector.to_numpy()

    assert len(result) == expected_size
    assert len(result[result == 1]) == len(true_matches)


def test_feature_matrix(
    attr_comparison_rl, left_source, right_source, comparison_config
):
    expected_size = len(left_source) * len(right_source)
    result = attr_comparison_rl.feature_matrix

    assert result.shape == (expected_size, len(comparison_config))


@pytest.mark.parametrize(
    "pm_comparison_rl, expected_col_count",
    [(2, 8), (3, 27)],
    indirect=["pm_comparison_rl"],
)
def test_pattern_matching_feature_matrix(
    pm_comparison_rl, left_source, right_source, expected_col_count
):
    expected_size = len(left_source) * len(right_source)
    result = pm_comparison_rl.feature_matrix

    assert result.shape == (expected_size, expected_col_count)
