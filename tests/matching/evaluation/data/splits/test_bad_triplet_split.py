import logging
from itertools import pairwise

import polars as pl
import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference, EntityReferenceIdentifier

from matchescu.matching.evaluation.data.splits import (
    BadTripletSplitGenerator,
    SplitGenerator,
)

RATIO = {"train": 3, "valid": 1, "test": 1}


def _ref(label: str, source: str = "source") -> EntityReference:
    return EntityReference(EntityReferenceIdentifier(label, source), {"name": label})


def _build(clusters, label_two):
    """Chain label-1 pairs inside each cluster; add the given label-2 pairs."""
    id_table, cluster_gt = InMemoryIdTable(), {}
    for cluster, labels in clusters.items():
        for label in labels:
            ref = _ref(label)
            id_table.put(ref)
            cluster_gt[ref.id] = cluster
    refs = {ref.id.label: ref.id for ref in id_table}
    matcher_gt = {
        (refs[left], refs[right]): 1
        for labels in clusters.values()
        for left, right in pairwise(labels)
    }
    matcher_gt.update({(refs[left], refs[right]): 2 for left, right in label_two})
    return id_table, cluster_gt, matcher_gt


def _contested_data():
    clusters = {i: [f"r{i}a", f"r{i}b"] for i in range(10)}
    for home in (0, 2, 4, 6, 8):
        clusters[home].append(f"x{home}")
    # x0 and x8 both target cluster 1; the other bridges are independent
    label_two = [
        ("r1a", "x0"),
        ("r3a", "x2"),
        ("r5a", "x4"),
        ("r7a", "x6"),
        ("r1b", "x8"),
    ]
    return clusters, label_two


def _quota_data():
    clusters = {i: [f"r{i}a", f"r{i}b"] for i in range(30)}
    for home in range(15):
        clusters[home].append(f"x{home}")
    label_two = [(f"r{15 + i}a", f"x{i}") for i in range(15)]
    return clusters, label_two


def _frame_refs(frame: pl.DataFrame) -> set[EntityReferenceIdentifier]:
    rows = list(frame.iter_rows(named=True))
    return {EntityReferenceIdentifier(r["left_id"], r["left_source"]) for r in rows} | {
        EntityReferenceIdentifier(r["right_id"], r["right_source"]) for r in rows
    }


def _label_two(frame: pl.DataFrame):
    return [
        (
            EntityReferenceIdentifier(r["left_id"], r["left_source"]),
            EntityReferenceIdentifier(r["right_id"], r["right_source"]),
        )
        for r in frame.iter_rows(named=True)
        if r["label"] == 2
    ]


def _bridge_clusters(bridges, cluster_gt):
    return [
        cluster_gt[ref] for pairs in bridges.values() for pair in pairs for ref in pair
    ]


@pytest.fixture
def split_data():
    clusters = {
        0: ["a0", "a1", "x"],
        1: ["b0", "b1", "w"],
        2: ["c0", "c1", "y"],
        3: ["d0", "d1"],
        4: ["e0", "e1"],
        5: ["f0", "f1"],
    }
    id_table = InMemoryIdTable()
    cluster_gt = {}
    for cluster, labels in clusters.items():
        for label in labels:
            ref = _ref(label)
            id_table.put(ref)
            cluster_gt[ref.id] = cluster
    refs = {ref.id.label: ref.id for ref in id_table}
    label_one = [
        ("a0", "a1"),
        ("a1", "x"),
        ("b0", "b1"),
        ("b0", "w"),
        ("c0", "c1"),
        ("c1", "y"),
        ("d0", "d1"),
        ("e0", "e1"),
        ("f0", "f1"),
    ]
    # star: every bridge touches cluster 1, so only one can be kept
    label_two = [("b0", "x"), ("b1", "y"), ("f0", "w")]
    matcher_gt = {
        **{(refs[left], refs[right]): 1 for left, right in label_one},
        **{(refs[left], refs[right]): 2 for left, right in label_two},
    }
    return id_table, cluster_gt, matcher_gt, clusters


def test_bad_triplet_split_keeps_clusters_disjoint(split_data, caplog):
    id_table, cluster_gt, matcher_gt, _ = split_data
    generator = BadTripletSplitGenerator(
        split_ratio=RATIO,
        log=logging.getLogger("bad-triplet-test"),
    ).load(id_table, cluster_gt, matcher_gt)

    with caplog.at_level(logging.WARNING, logger="bad-triplet-test"):
        generator.generate()

    split_records = {
        name: _frame_refs(frame) for name, frame in generator._splits.items()
    }
    split_clusters = {
        name: {cluster_gt[ref_id] for ref_id in records if ref_id in cluster_gt}
        for name, records in split_records.items()
    }

    all_records = list(split_records.values())
    assert sum(len(records) for records in all_records) == len(
        set().union(*all_records)
    )

    bridges = {name: _label_two(frame) for name, frame in generator._splits.items()}
    for name, pairs in bridges.items():
        for left, right in pairs:
            assert cluster_gt[left] in split_clusters[name]
            assert cluster_gt[right] in split_clusters[name]

    used = _bridge_clusters(bridges, cluster_gt)
    assert len(used) == len(set(used))
    assert {name: len(pairs) for name, pairs in bridges.items()} == {
        "train": 1,
        "valid": 0,
        "test": 0,
    }
    assert caplog.text.count("dropping bridge") == 2
    assert "splits without bridges" in caplog.text


def test_contested_winner_goes_to_train_and_eval_splits_get_bridges(caplog):
    id_table, cluster_gt, matcher_gt = _build(*_contested_data())
    generator = BadTripletSplitGenerator(
        split_ratio=RATIO,
        log=logging.getLogger("contested-test"),
    ).load(id_table, cluster_gt, matcher_gt)

    with caplog.at_level(logging.WARNING, logger="contested-test"):
        generator.generate()

    bridges = {name: _label_two(frame) for name, frame in generator._splits.items()}
    assert all(bridges.values())
    assert sum(map(len, bridges.values())) == 4
    used = _bridge_clusters(bridges, cluster_gt)
    assert len(used) == len(set(used))
    assert any(cluster_gt[target] == 1 for target, _ in bridges["train"])
    assert caplog.text.count("dropping bridge") == 1
    assert len(generator.orphan_records) == 1

    all_records = [_frame_refs(frame) for frame in generator._splits.values()]
    assert sum(map(len, all_records)) == len(set().union(*all_records))


def test_drop_orphans_removes_orphaned_bridge_records():
    id_table, cluster_gt, matcher_gt = _build(*_contested_data())
    generator = (
        BadTripletSplitGenerator(split_ratio=RATIO, drop_orphans=True)
        .load(id_table, cluster_gt, matcher_gt)
        .generate()
    )

    (orphan,) = generator.orphan_records
    assert all(orphan not in _frame_refs(f) for f in generator._splits.values())


def test_chunks_fill_eval_splits_first_when_items_are_scarce():
    generator = BadTripletSplitGenerator(split_ratio=RATIO)
    assert generator._chunks([], [3, 1, 1]) == [[], [], []]
    assert generator._chunks(["a", "b"], [3, 1, 1]) == [[], ["a"], ["b"]]
    assert generator._chunks(list("abcde"), [3, 1, 1]) == [
        ["a", "b", "c"],
        ["d"],
        ["e"],
    ]


def test_chunks_fill_train_first_when_bridge_units_are_scarce():
    generator = BadTripletSplitGenerator(split_ratio=RATIO)
    assert generator._chunks([], [3, 1, 1], eval_first=False) == [[], [], []]
    assert generator._chunks(["a", "b"], [3, 1, 1], eval_first=False) == [
        ["a", "b"],
        [],
        [],
    ]
    assert generator._chunks(["a"], [3, 1, 1], eval_first=False) == [["a"], [], []]
    assert generator._chunks(list("abcde"), [3, 1, 1], eval_first=False) == [
        ["a", "b", "c"],
        ["d"],
        ["e"],
    ]


def test_bridge_units_and_positives_are_chunked_by_ratio():
    id_table, cluster_gt, matcher_gt = _build(*_quota_data())
    generator = (
        BadTripletSplitGenerator(split_ratio=RATIO)
        .load(id_table, cluster_gt, matcher_gt)
        .generate()
    )

    bridges = {name: _label_two(frame) for name, frame in generator._splits.items()}
    assert {name: len(pairs) for name, pairs in bridges.items()} == {
        "train": 9,
        "valid": 3,
        "test": 3,
    }

    positives = {
        name: sum(1 for r in frame.iter_rows(named=True) if r["label"] > 0)
        for name, frame in generator._splits.items()
    }
    assert positives == {"train": 27, "valid": 9, "test": 9}

    used = _bridge_clusters(bridges, cluster_gt)
    assert len(used) == len(set(used))


def test_bad_triplet_split_writes_configured_file_names(split_data, tmp_path):
    id_table, cluster_gt, matcher_gt, _ = split_data
    generator = (
        BadTripletSplitGenerator(split_ratio=RATIO)
        .load(id_table, cluster_gt, matcher_gt)
        .generate()
    )

    generator.save(
        tmp_path,
        file_names={
            "train": "train-trial.csv",
            "valid": "valid-trial.csv",
            "test": "test-trial.csv",
        },
    )

    assert (tmp_path / "train-trial.csv").exists()
    assert (tmp_path / "valid-trial.csv").exists()
    assert (tmp_path / "test-trial.csv").exists()


def test_reversed_positive_is_a_warning_and_a_negative(caplog):
    left, right = _ref("left"), _ref("right")
    id_table = InMemoryIdTable().put(left).put(right)
    generator = SplitGenerator(log=logging.getLogger("reverse-test"))
    generator._id_table = id_table
    generator._cluster_gt = {left.id: 0, right.id: 0}
    generator._matcher_gt = {(right.id, left.id): 1}
    generator._splits = {
        "train": pl.DataFrame(
            {
                "left_id": [left.id.label],
                "left_source": [left.id.source],
                "right_id": [right.id.label],
                "right_source": [right.id.source],
            }
        )
    }

    with caplog.at_level(logging.WARNING, logger="reverse-test"):
        splits = generator.get_splits()

    assert "reverse order" in caplog.text
    assert splits["train"].matcher_labels == {}


def test_clamp_rejects_fewer_than_minimum_pairs():
    with pytest.raises(ValueError, match="MIN_PER_CLASS"):
        SplitGenerator._clamp(3, 2)


def test_split_counts_preserve_ratio():
    assert SplitGenerator._split_counts(10, [3, 1, 1]) == [6, 2, 2]
