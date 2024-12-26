import itertools

import polars as pl
import pytest

from matchescu.matching.entity_reference import (
    RawComparison,
    EntityReferenceComparisonConfig,
)
from matchescu.matching.extraction import CsvDataSource, Traits
from matchescu.matching.ml.datasets import DeduplicationDataSet


@pytest.fixture(scope="module")
def dataset_dir(data_dir):
    return data_dir / "cora"


@pytest.fixture(scope="module")
def cora(dataset_dir):
    traits = list(Traits().int([0]).string([2, 3, 5, 7]))
    return CsvDataSource("cora", traits, has_header=False).read_csv(
        dataset_dir / "cora.csv"
    )


@pytest.fixture(scope="module")
def cora_ground_truth(dataset_dir):
    df = pl.read_csv(dataset_dir / "cora.csv", has_header=False, ignore_errors=True)
    groups = df.select(pl.col("column_1", "column_3")).group_by(pl.col("column_3"))
    result = set()
    for group, items in groups:
        item_ids = (val[0] for val in items.select(pl.col("column_1")).iter_rows())
        id_combinations = itertools.combinations(item_ids, 2)
        for item_pair in id_combinations:
            result.add(item_pair)
    return result


@pytest.fixture(scope="module")
def comparison_config() -> EntityReferenceComparisonConfig:
    return RawComparison().levenshtein("name", 2, 2)


@pytest.fixture(scope="module")
def dataset(cora, cora_ground_truth, comparison_config) -> DeduplicationDataSet:
    result = DeduplicationDataSet(cora, cora_ground_truth).attr_compare(
        comparison_config
    )
    result.cross_sources()
    return result


def test_feature_matrix(cora, dataset):
    n = len(cora)

    assert len(dataset.feature_matrix) == (n * (n - 1)) / 2


def test_target_vector(cora_ground_truth, dataset):
    y = dataset.target_vector.to_numpy()

    assert len(y[y == 1]) == len(cora_ground_truth)
