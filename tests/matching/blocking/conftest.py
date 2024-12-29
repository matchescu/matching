import csv
from typing import Hashable

import pytest

from matchescu.matching.extraction import Traits, ListDataSource, CsvDataSource
from matchescu.typing import EntityReference, Record, DataSource


@pytest.fixture
def ds_traits() -> Traits:
    return Traits().string([0, 1]).int([2])


@pytest.fixture
def ds1(request, ds_traits) -> ListDataSource:
    ds1 = ListDataSource("ds1", ds_traits).append(
        [
            "Netgear ProSafe 5 Port 10/100 Desktop Switch - FS105",
            "Netgear ProSafe 5 Port 10/100 Desktop Switch - FS105/ 5 Auto Speed-Sensing 10/100 UTP Ports/ Embedded Memory",
            33053,
        ]
    )
    if hasattr(request, "param") and isinstance(request.param, (list, tuple, set)):
        ds1.append(request.param)
    return ds1


@pytest.fixture
def ds2(ds_traits) -> ListDataSource:
    return ListDataSource("ds2", ds_traits).append(
        [
            "Netgear ProSafe FS105 Ethernet Switch - FS105NA",
            "NETGEAR FS105 Prosafe 5 Port 10/100 Desktop Switch",
            10221960,
        ]
    )


@pytest.fixture
def ds_identifier():
    def _id(x: EntityReference) -> Hashable:
        return x[2]

    return _id


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
def abt_buy_perfect_mapping(data_dir) -> set[tuple[int, int]]:
    with open(data_dir / "abt-buy" / "abt_buy_perfectMapping.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        return set((id_abt, id_buy) for (id_abt, id_buy) in reader)


@pytest.fixture
def abt_buy_identifier():
    def _id(x: EntityReference) -> Hashable:
        return x[0]
    return _id
