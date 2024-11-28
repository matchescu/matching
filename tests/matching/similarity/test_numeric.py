import pytest

from matchescu.matching.similarity._numeric import BoundedNumericDifferenceSimilarity


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
