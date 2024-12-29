from typing import Hashable

import pytest

from matchescu.matching.extraction import Traits, ListDataSource
from matchescu.typing import EntityReference


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
