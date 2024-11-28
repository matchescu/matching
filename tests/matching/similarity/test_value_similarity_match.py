import pytest

from matchescu.matching.similarity import Similarity
from matchescu.matching.attribute import (
    TernaryResult,
    TernarySimilarityMatchOnThreshold,
    BinarySimilarityMatchOnThreshold,
    BinaryResult,
)


@pytest.mark.parametrize(
    "match_strategy,expected",
    [
        (TernarySimilarityMatchOnThreshold, TernaryResult.NoComparisonData),
        (BinarySimilarityMatchOnThreshold, BinaryResult.Negative),
    ],
)
def test_no_comparison_data(match_strategy, expected, similarity_stub):
    is_match = match_strategy(similarity_stub)

    assert is_match(None, None) == expected


@pytest.mark.parametrize(
    "match_strategy, similarity_stub, threshold, expected",
    [
        (TernarySimilarityMatchOnThreshold, 0, 0.01, TernaryResult.NonMatch),
        (TernarySimilarityMatchOnThreshold, 1, 1, TernaryResult.Match),
        (TernarySimilarityMatchOnThreshold, 0.5, 0.49, TernaryResult.Match),
        (BinarySimilarityMatchOnThreshold, 0, 0.01, BinaryResult.Negative),
        (BinarySimilarityMatchOnThreshold, 1, 1, BinaryResult.Positive),
        (BinarySimilarityMatchOnThreshold, 0.5, 0.49, BinaryResult.Positive),
    ],
    indirect=["similarity_stub"],
)
def test_value_similarity_match(match_strategy, similarity_stub, threshold, expected):
    is_match = match_strategy(similarity_stub, threshold)

    assert is_match("can pass any value", "with stubbed similarity") == expected
