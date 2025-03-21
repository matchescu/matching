import pytest

from matchescu.matching.blocking._filter import JaccardSimilarityFilter
from matchescu.typing import EntityReferenceIdentifier


def _id(label: int, source: str):
    return EntityReferenceIdentifier(label, source)

@pytest.fixture
def jaccard_filter(abt_buy_id_table, request):
    min_sim = request.param if hasattr(request, 'param') and isinstance(request.param, float) else 0.5
    return JaccardSimilarityFilter(abt_buy_id_table, min_sim)


@pytest.mark.parametrize(
    "jaccard_filter,ref1,ref2,expected",
    [
        (5/28, _id(38477, "Abt"), _id(10011646, "Buy"), True),
        (6/28, _id(38477, "Abt"), _id(10011646, "Buy"), False),
    ],
    indirect=["jaccard_filter"],
)
def test_jaccard_filter_removes_items(
    jaccard_filter, ref1, ref2, expected
):
    assert jaccard_filter(ref1, ref2) == expected
