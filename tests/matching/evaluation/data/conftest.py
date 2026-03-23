import pytest

from matchescu.extraction import Traits
from matchescu.typing import EntityReferenceIdentifier as RefId


@pytest.fixture(scope="package")
def ag_dir(data_dir):
    return data_dir / "amazon_google_exp_data"


@pytest.fixture(scope="package")
def ag_traits():
    return Traits().string(["title", "manufacturer"]).currency(["price"])


@pytest.fixture(scope="package")
def ag_id_factory():
    def _(record, source):
        return RefId(label=record[0]["id"], source=source)

    return _


@pytest.fixture(scope="package")
def affils_dir(data_dir):
    return data_dir / "affiliationstrings"


@pytest.fixture(scope="package")
def affils_traits():
    return Traits().string(["affil1"])


@pytest.fixture(scope="package")
def affils_id_col():
    return "id1"
