from matchescu.data import EntityReferenceExtraction
from matchescu.matching.blocking import BlockEngine
from matchescu.matching.extraction import ListDataSource, Traits


def test_block_engine_lists_data_source_names():
    traits = Traits().string([0, 1]).int([2])
    ds1 = ListDataSource("ds1", traits).append(["abc", "def ghi", "21"])
    ds2 = ListDataSource("ds2", traits).append(["abc", "def jkl", "25"])
    engine = BlockEngine(
        [
            EntityReferenceExtraction(ds1, lambda x: x[2]),
            EntityReferenceExtraction(ds2, lambda x: x[2]),
        ]
    ).jaccard_blocks(1, 0.4)

    source_names = list(engine.list_source_names())

    assert ["ds1", "ds2"] == source_names


def test_block_engine_lists_unique_data_source_names():
    traits = Traits().string([0, 1]).int([2])
    ds1 = ListDataSource("ds1", traits).append(["abc", "def ghi", "21"])
    ds2 = ListDataSource("ds2", traits).append(["abc", "def jkl", "25"])
    engine = (
        BlockEngine(
            [
                EntityReferenceExtraction(ds1, lambda x: x[2]),
                EntityReferenceExtraction(ds2, lambda x: x[2]),
            ]
        )
        .jaccard_blocks(1, 0.4)
        .jaccard_blocks(0, 0.6)
    )
    # the second call to jaccard_blocks duplicates the data source names

    source_names = list(engine.list_source_names())

    assert ["ds1", "ds2"] == source_names


def test_block_engine_returns_only_filtered_block_records():
    traits = Traits().string([0, 1]).int([2])
    ds1 = ListDataSource("ds1", traits).extend(
        [
            ["abc", "def ghi", "21"],
            ["abc", "def jkl", "22"],
        ]
    )
    ds2 = ListDataSource("ds2", traits).append(["abc", "def jkl", "25"])
    engine = (
        BlockEngine(
            [
                EntityReferenceExtraction(ds1, lambda x: x[2]),
                EntityReferenceExtraction(ds2, lambda x: x[2]),
            ]
        )
        .jaccard_blocks(1, 0.4)
        .cross_sources_filter()
    )
    # the second call to jaccard_blocks duplicates the data source names

    ds = engine.create_data_sources()

    assert len(ds) == 2
    assert len(ds["ds1"]) == 1
    assert len(ds["ds2"]) == 1
