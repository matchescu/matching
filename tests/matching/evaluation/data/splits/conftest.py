import logging
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching.evaluation.data.splits import BadTripletSplitGenerator

PAIR_SOURCE = "pairs"
GENERATED_SOURCE = "ambiguity-generator"
SOURCELESS_BRIDGE = RefId(108, GENERATED_SOURCE)
SECOND_BRIDGE = RefId(109, GENERATED_SOURCE)
SHARED_HOME = 1
SHARED_HOME_MEMBER = RefId(1, "synthetic")
SECOND_BRIDGE_TARGET = RefId(39, "synthetic")

SYNTHETIC = "bad_triplet_synthetic"
SEED = 41
PREFIX = "split"


def _rows(path: Path) -> list[dict]:
    return list(pl.read_csv(path, infer_schema_length=0).iter_rows(named=True))


def _dataset_rows(directory: Path, kind: str) -> list[dict]:
    return _rows(directory / f"{directory.name}_{kind}.csv")


def _ref_id(row: dict, side: str = "") -> RefId:
    return RefId(int(row[f"{side}id"]), row[f"{side}source"])


def _id_table(directory: Path) -> InMemoryIdTable:
    id_table = InMemoryIdTable()
    for row in _dataset_rows(directory, "ids"):
        id_table.put(EntityReference(_ref_id(row), row))
    return id_table


def _cluster_gt(directory: Path) -> dict[RefId, int]:
    rows = _dataset_rows(directory, "cluster_mapping")
    return {_ref_id(row): int(row["cluster_id"]) for row in rows}


def _matcher_gt(directory: Path) -> dict[tuple[RefId, RefId], int]:
    rows = _dataset_rows(directory, "mapping")
    return {
        (_ref_id(row, "left_"), _ref_id(row, "right_")): int(row["label"])
        for row in rows
    }


def _labelled_rows(path: Path) -> list[tuple[RefId, RefId, int]]:
    return [
        (_ref_id(row, "left_"), _ref_id(row, "right_"), int(row["label"]))
        for row in _rows(path)
    ]


@pytest.fixture(scope="session")
def dataset(request, data_dir):
    directory = data_dir / getattr(request, "param", SYNTHETIC)
    return _id_table(directory), _cluster_gt(directory), _matcher_gt(directory)


@pytest.fixture(scope="session")
def split_ratio():
    return {"train": 3, "dev": 1, "test": 1}


@pytest.fixture
def logger(caplog):
    log = logging.getLogger("bad-triplet-test")
    caplog.set_level(logging.INFO, logger=log.name)
    return log


@pytest.fixture
def generator_class():
    return BadTripletSplitGenerator


@pytest.fixture
def make_generator(generator_class, split_ratio, logger):
    def _make(data, **kwargs):
        generator = generator_class(
            split_ratio=split_ratio, seed=SEED, log=logger, **kwargs
        )
        return generator.load(*data)

    return _make


@pytest.fixture
def generator_options(request):
    return getattr(request, "param", {})


@pytest.fixture
def generated(make_generator, dataset, generator_options):
    return make_generator(dataset, **generator_options).generate()


@pytest.fixture
def saved_dir(generated, tmp_path):
    generated.save(tmp_path, prefix=PREFIX)
    return tmp_path


@pytest.fixture
def split_rows(saved_dir, split_ratio):
    return {
        name: _labelled_rows(saved_dir / f"{PREFIX}_{name}.csv") for name in split_ratio
    }


@pytest.fixture
def meta_paths(saved_dir, split_ratio):
    return {name: saved_dir / f"{PREFIX}_{name}_meta.csv" for name in split_ratio}


@pytest.fixture
def meta_rows(meta_paths):
    return {name: _rows(path) for name, path in meta_paths.items()}


@pytest.fixture
def category_counts(meta_rows):
    return {
        name: Counter(row["category"] for row in rows)
        for name, rows in meta_rows.items()
    }


@pytest.fixture
def source_counts(meta_rows):
    rows = (row for split in meta_rows.values() for row in split)
    return Counter(
        int(row["bridge_id"]) for row in rows if row["category"] == "triplet_source"
    )


@pytest.fixture
def label_one_counts(split_rows):
    return {
        name: sum(1 for *_, label in rows if label == 1)
        for name, rows in split_rows.items()
    }


@pytest.fixture
def cluster_splits(split_rows, dataset):
    _, cluster_gt, _ = dataset
    result = defaultdict(set)
    for name, rows in split_rows.items():
        for left, right, _ in rows:
            result[cluster_gt[left]].add(name)
            result[cluster_gt[right]].add(name)
    return result


@pytest.fixture
def label_two_rows(split_rows):
    return {
        name: {(left, right) for left, right, label in rows if label == 2}
        for name, rows in split_rows.items()
    }


@pytest.fixture
def label_two_truth(dataset):
    _, _, matcher_gt = dataset
    return {pair for pair, label in matcher_gt.items() if label == 2}


@pytest.fixture
def footprints(dataset, label_two_truth):
    _, cluster_gt, _ = dataset
    result = defaultdict(set)
    for target, bridge in label_two_truth:
        result[bridge] |= {cluster_gt[target], cluster_gt[bridge]}
    return result


@pytest.fixture
def footprint_splits(footprints, cluster_splits):
    return {
        bridge: set().union(*(cluster_splits[cluster] for cluster in clusters))
        for bridge, clusters in footprints.items()
    }


@pytest.fixture
def target_clusters(dataset, label_two_truth):
    _, cluster_gt, _ = dataset
    result = defaultdict(set)
    for target, bridge in label_two_truth:
        result[bridge].add(cluster_gt[target])
    return result


@pytest.fixture
def negatives(split_rows):
    return [
        (left, right)
        for rows in split_rows.values()
        for left, right, label in rows
        if label == 0
    ]


@pytest.fixture
def target_negatives(negatives, dataset, target_clusters):
    _, cluster_gt, _ = dataset
    return [
        (left, right)
        for left, right in negatives
        if cluster_gt[right] in target_clusters.get(left, set())
        or cluster_gt[left] in target_clusters.get(right, set())
    ]


@pytest.fixture
def shared_home_data(dataset):
    id_table, cluster_gt, matcher_gt = dataset
    return (
        id_table,
        {**cluster_gt, SECOND_BRIDGE: SHARED_HOME},
        {
            **matcher_gt,
            (SHARED_HOME_MEMBER, SECOND_BRIDGE): 1,
            (SECOND_BRIDGE_TARGET, SECOND_BRIDGE): 2,
        },
    )


@pytest.fixture
def sourceless_data(dataset):
    id_table, cluster_gt, matcher_gt = dataset
    kept = {
        pair: label
        for pair, label in matcher_gt.items()
        if label != 1 or SOURCELESS_BRIDGE not in pair
    }
    return id_table, cluster_gt, kept


@pytest.fixture
def select_by_source():
    def _select(matcher_gt):
        refs = (ref for pair in matcher_gt for ref in pair)
        return frozenset(ref for ref in refs if ref.source == GENERATED_SOURCE)

    return _select


@pytest.fixture
def select_nothing():
    def _select(_):
        return frozenset()

    return _select


@pytest.fixture
def label_three_data(dataset):
    id_table, cluster_gt, matcher_gt = dataset
    first_pair = next(iter(matcher_gt))
    return id_table, cluster_gt, {**matcher_gt, first_pair: 3}


@pytest.fixture
def make_pair_data():
    def _make(cluster_count, label_two_count):
        refs = [
            [RefId(2 * c + i, PAIR_SOURCE) for i in range(2)]
            for c in range(cluster_count)
        ]
        id_table = InMemoryIdTable()
        for ref in (ref for pair in refs for ref in pair):
            id_table.put(EntityReference(ref, {"id": ref.label}))
        cluster_gt = {ref: c for c, pair in enumerate(refs) for ref in pair}
        matcher_gt = {tuple(pair): 1 for pair in refs}
        matcher_gt.update(
            {(refs[2 * c][0], refs[2 * c + 1][1]): 2 for c in range(label_two_count)}
        )
        return id_table, cluster_gt, matcher_gt

    return _make
