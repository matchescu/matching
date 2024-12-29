import polars as pl
import pytest

from matchescu.matching.extraction import Traits, CsvDataSource


@pytest.fixture
def dataset_dir(data_dir):
    return data_dir / "abt-buy"


@pytest.fixture
def left_source(dataset_dir):
    traits = list(Traits().int([0]).string([1, 2]).currency([3]))
    return CsvDataSource("abt", traits).read_csv(dataset_dir / "Abt.csv")


@pytest.fixture
def right_source(dataset_dir):
    traits = list(Traits().int([0]).string([1, 2, 3]).currency([4]))
    return CsvDataSource("buy", traits).read_csv(dataset_dir / "Buy.csv")


@pytest.fixture
def true_matches(dataset_dir):
    perfect_mapping_path = dataset_dir / "abt_buy_perfectMapping.csv"
    return set(pl.read_csv(perfect_mapping_path, ignore_errors=True).iter_rows())
