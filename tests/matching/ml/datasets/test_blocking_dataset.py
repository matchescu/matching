import pytest

from matchescu.matching.blocking import BlockEngine
from matchescu.matching.entity_reference import RawComparison
from matchescu.matching.ml.datasets._blocking import BlockDataSet


@pytest.fixture
def block_engine(left_source, right_source):
    engine = (
        BlockEngine()
        .add_source(left_source, lambda x: x[0])
        .add_source(right_source, lambda x: x[0])
        .tf_idf(0.3)
    )
    engine.update_candidate_pairs(False)
    engine.filter_candidates_jaccard(0.15)
    return engine


@pytest.fixture
def comparison_config():
    return (
        RawComparison()
        .levenshtein_distance("name", 1, 1)
        .levenshtein_distance("description", 2, 2)
        .diff("price", 3, 4)
    )


def test_dataset_has_expected_size(block_engine, true_matches, comparison_config):
    ds = BlockDataSet(
        block_engine, true_matches, lambda x: x[0], lambda x: x[0]
    ).attr_compare(comparison_config)

    ds.cross_sources()

    assert ds.feature_matrix.shape == (
        block_engine.candidate_count,
        len(comparison_config),
    )
    assert ds.target_vector.shape == (block_engine.candidate_count,)
