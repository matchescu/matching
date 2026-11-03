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


@pytest.fixture
def binary_dataset(dataset):
    id_table, cluster_gt, matcher_gt = dataset
    return (
        id_table,
        cluster_gt,
        {pair: 1 if label == 2 else label for pair, label in matcher_gt.items()},
    )


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
    def _make(data, ratios=None, **kwargs):
        generator = generator_class(
            split_ratio=ratios or split_ratio, seed=SEED, log=logger, **kwargs
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
def link_counts(category_counts):
    return {name: counts["triplet_target"] for name, counts in category_counts.items()}


@pytest.fixture
def bridges(dataset):
    id_table, _, _ = dataset
    return frozenset(ref.id for ref in id_table if ref.id.source == GENERATED_SOURCE)


@pytest.fixture
def link_truth(dataset, bridges):
    _, cluster_gt, matcher_gt = dataset
    truth: dict[RefId, set] = defaultdict(set)
    for pair, label in matcher_gt.items():
        if not label:
            continue
        left, right = pair
        if (left in bridges) == (right in bridges):
            continue
        bridge = left if left in bridges else right
        target = right if left == bridge else left
        if cluster_gt.get(target) == cluster_gt.get(bridge):
            continue
        truth[bridge].add((target, bridge))
    return {bridge: frozenset(pairs) for bridge, pairs in truth.items()}


@pytest.fixture
def source_truth(dataset, bridges):
    _, cluster_gt, matcher_gt = dataset
    truth: dict[RefId, set] = defaultdict(set)
    for pair, label in matcher_gt.items():
        if not label:
            continue
        left, right = pair
        if (left in bridges) == (right in bridges):
            continue
        bridge = left if left in bridges else right
        home = right if left == bridge else left
        if cluster_gt.get(home) != cluster_gt.get(bridge):
            continue
        truth[bridge].add((home, bridge))
    return {bridge: frozenset(pairs) for bridge, pairs in truth.items()}


@pytest.fixture
def footprints(link_truth, dataset):
    _, cluster_gt, _ = dataset
    return {
        bridge: frozenset(
            {cluster_gt[target] for target, _ in pairs} | {cluster_gt[bridge]}
        )
        for bridge, pairs in link_truth.items()
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
def footprint_splits(footprints, cluster_splits):
    return {
        bridge: set().union(*(cluster_splits[cluster] for cluster in clusters))
        for bridge, clusters in footprints.items()
    }


@pytest.fixture
def target_clusters(dataset, link_truth):
    _, cluster_gt, _ = dataset
    return {
        bridge: frozenset(cluster_gt[target] for target, _ in pairs)
        for bridge, pairs in link_truth.items()
    }


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
def repeated_rows(split_rows):
    rows = Counter(
        frozenset((left, right))
        for split in split_rows.values()
        for left, right, _ in split
    )
    return [pair for pair, count in rows.items() if count > 1]


@pytest.fixture
def select_by_source():
    def _select(ref):
        return ref.source == GENERATED_SOURCE

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
