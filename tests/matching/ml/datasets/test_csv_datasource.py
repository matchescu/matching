import pytest

from matchescu.matching.extraction import CsvDataSource, Traits


@pytest.fixture
def csv_path(data_dir):
    return data_dir / "table.csv"


@pytest.fixture
def traits():
    return list(Traits().string([0, 1, 2, 3]).int([4]))


@pytest.fixture
def data_source(traits):
    return CsvDataSource(name="test", traits=traits)


def test_read_csv(data_source, csv_path):
    ds = data_source.read_csv(csv_path)

    assert len(ds) > 0
