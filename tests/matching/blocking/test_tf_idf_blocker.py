import pytest

from matchescu.matching.blocking import TfIdfBlocker


@pytest.fixture
def tf_idf_blocker(abt_buy_id_table, request):
    min_score = request.param if hasattr(request, "param") and isinstance(request.param, float) else 0.1
    return TfIdfBlocker(abt_buy_id_table, min_score)


def test_abt_buy_blocking_no_data_loss(tf_idf_blocker, abt, buy):
    blocks = list(tf_idf_blocker())
    all_ids = set(identifier for block in blocks for identifier in block)
    assert len(all_ids) == len(abt) + len(buy)

