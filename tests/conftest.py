import polars
import pytest

from pathlib import Path


TEST_DIR = Path(__file__).parent


def _load_data(filename: str, headers: bool) -> polars.DataFrame:
    with open(TEST_DIR / "data" / filename, "r") as csv_file:
        return polars.read_csv(csv_file, header=1 if headers else 0)


@pytest.fixture(scope="session")
def data_dir():
    return TEST_DIR / "data"


@pytest.fixture
def subsample_a():
    return _load_data("subsample_a.csv", False)


@pytest.fixture
def subsample_b():
    return _load_data("subsample_b.csv", False)


@pytest.fixture
def sub_table_a():
    return _load_data("subtable_a.csv", True)


@pytest.fixture
def sub_table_b():
    return _load_data("subtable_b.csv", True)
