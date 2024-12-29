import pytest

from matchescu.matching.blocking import BlockEngine


@pytest.fixture
def engine(ds1, ds2, ds_identifier):
    return (
        BlockEngine()
        .add_source(ds1, lambda x: x[2])
        .add_source(ds2, lambda x: x[2])
        .jaccard_canopy(1, 0.4)
    )


def test_block_engine_lists_data_source_names(engine):
    source_names = list(engine.list_source_names())

    assert ["ds1", "ds2"] == source_names


def test_block_engine_lists_unique_data_source_names(engine):
    # create duplicate blocks
    engine.jaccard_canopy(0, 0.6)

    source_names = list(engine.list_source_names())

    assert ["ds1", "ds2"] == source_names
