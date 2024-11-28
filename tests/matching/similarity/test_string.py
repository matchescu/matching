import pytest

from matchescu.matching.similarity import Jaro, JaroWinkler, Jaccard, Levenshtein


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("abc", "def", 0),
        ("a", "", 0),
        ("", "a", 0),
        ("land", "dru", 0),  # this test case showcases an inconvenient fact
        ("andy", "warhol", 0.17),
        ("bonanza", "andy", 0.29),
        ("nand", "andy", 0.5),
        ("abc", "ab", 0.67),
        ("ab", "abc", 0.67),
        ("abc", "ac", 0.67),
        ("constantin", "constantinescu", 0.71),
    ],
)
def test_levenshtein(a, b, expected):
    similarity = Levenshtein()

    assert 0 <= similarity(a, b) <= 1
    assert similarity(a, b) == expected


@pytest.mark.parametrize("factory_method", [Levenshtein, Jaro, JaroWinkler, Jaccard])
@pytest.mark.parametrize(
    "a, b",
    [
        (None, None),
        (None, ""),
        ("", None),
        ("", "a"),
        ("a", ""),
        ("abc", "def"),
        ("abc", "cde"),
        ("abc", "ade"),
        ("abc", "dbe"),
        ("abcde", "cd"),
        ("abc", "abc"),
    ],
)
def test_string_similarity_returns_value_in_expected_interval(factory_method, a, b):
    similarity = factory_method()

    assert 0 <= similarity(a, b) <= 1
