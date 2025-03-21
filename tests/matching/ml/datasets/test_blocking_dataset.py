import pytest

from matchescu.matching.blocking import Blocker, TfIdfBlocker
from matchescu.matching.entity_reference import RawComparison
from matchescu.matching.ml.datasets._blocking import BlockDataSet


@pytest.fixture
def blocker(abt_buy_id_table):
    return TfIdfBlocker(abt_buy_id_table)


@pytest.fixture
def comparison_config():
    return (
        RawComparison()
        .levenshtein_distance("name", 1, 1)
        .levenshtein_distance("description", 2, 2)
        .diff("price", 3, 4)
    )


def test_dataset_has_expected_size(blocker, abt_buy_id_table, abt_buy_gt, comparison_config):
    ds = BlockDataSet(
        blocker, abt_buy_id_table, lambda x: x[0], lambda x: x[0], abt_buy_gt
    ).attr_compare(comparison_config)

    ds.cross_sources()

    assert ds.feature_matrix.shape == (30544, len(comparison_config))
    assert ds.target_vector.shape == (30544,)
