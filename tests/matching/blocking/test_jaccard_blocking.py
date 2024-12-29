import pytest

from matchescu.matching.blocking import BlockEngine


@pytest.fixture
def blocking_engine(abt, buy, abt_buy_identifier):
    return (
        BlockEngine()
        .add_source(abt, abt_buy_identifier)
        .add_source(buy, abt_buy_identifier)
    )


def test_canopy_clustering_by_name(blocking_engine):
    blocking_engine.jaccard_canopy(1, 0.5)

    assert blocking_engine.blocks


def test_multi_pass_canopy_clustering(blocking_engine):
    blocking_engine.jaccard_canopy(1, 0.5)
    num_blocks = len(blocking_engine.blocks)
    blocking_engine.jaccard_canopy(2, 0.5)

    assert len(blocking_engine.blocks) > num_blocks


def test_block_cross_source_filtering(blocking_engine):
    engine = blocking_engine.jaccard_canopy(1, 0.5)
    num_unfiltered_blocks = len(engine.blocks)
    engine = engine.only_multi_source_blocks()

    assert all(len(b.references) > 1 for b in engine.blocks)
    assert num_unfiltered_blocks > len(engine.blocks)
