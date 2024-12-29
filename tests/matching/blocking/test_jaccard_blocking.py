import csv
from typing import Hashable

import pytest

from matchescu.matching.blocking import BlockEngine
from matchescu.matching.extraction import CsvDataSource, Traits
from matchescu.typing import Record, DataSource, EntityReference


@pytest.fixture
def abt_traits() -> Traits:
    return Traits().int([0]).string([1, 2]).currency([3])


@pytest.fixture
def abt(data_dir, abt_traits) -> DataSource[Record]:
    ds = CsvDataSource("abt", abt_traits)
    ds.read_csv(data_dir / "abt-buy" / "Abt.csv")
    return ds


@pytest.fixture
def buy_traits() -> Traits:
    return Traits().int([0]).string([1, 2, 3]).currency([4])


@pytest.fixture
def buy(data_dir, buy_traits) -> DataSource[Record]:
    ds = CsvDataSource("buy", buy_traits)
    ds.read_csv(data_dir / "abt-buy" / "Buy.csv")
    return ds


@pytest.fixture
def perfect_mapping(data_dir) -> set[tuple[int, int]]:
    with open(data_dir / "abt-buy" / "abt_buy_perfectMapping.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        return set((id_abt, id_buy) for (id_abt, id_buy) in reader)


def _id(x: EntityReference) -> Hashable:
    return x[0]


@pytest.fixture
def blocking_engine(abt, buy):
    return BlockEngine().add_source(abt, _id).add_source(buy, _id)


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
