import pytest

from matchescu.blocking import TfIdfBlocker
from matchescu.comparison_filtering import (
    is_cross_source_comparison,
    JaccardSimilarityFilter,
)
from matchescu.csg import BinaryComparisonSpaceGenerator
from matchescu.matching.entity_reference import RawComparison
from matchescu.matching.ml.datasets._blocking import BlockDataSet
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture
def csg(abt_buy_id_table):
    return (
        BinaryComparisonSpaceGenerator()
        .add_blocker(TfIdfBlocker(abt_buy_id_table, 0.22))
        .add_filter(is_cross_source_comparison)
        .add_filter(JaccardSimilarityFilter(abt_buy_id_table, 0.25))
    )


@pytest.fixture
def comparison_config():
    return (
        RawComparison()
        .levenshtein_distance("name", 1, 1)
        .levenshtein_distance("description", 2, 2)
        .diff("price", 3, 4)
    )


def test_dataset_has_expected_size(
    csg, abt, buy, abt_buy_id_table, abt_buy_gt, comparison_config
):
    ds = BlockDataSet(
        csg,
        abt_buy_id_table,
        lambda x: EntityReferenceIdentifier(x[0], abt.name),
        lambda x: EntityReferenceIdentifier(x[0], buy.name),
        abt_buy_gt,
    ).attr_compare(comparison_config)

    ds.cross_sources()

    assert ds.feature_matrix.shape == (370, len(comparison_config))
    assert ds.target_vector.shape == (370,)
