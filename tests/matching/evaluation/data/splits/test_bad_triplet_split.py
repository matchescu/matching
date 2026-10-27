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
SEED = 42  # chosen so that the stranded fixture's bridge record is processed last


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


def _star_data():
    clusters = {
        0: ["a0", "a1", "x"],
        1: ["b0", "b1", "w"],
        2: ["c0", "c1", "y"],
        3: ["d0", "d1"],
        4: ["e0", "e1"],
        5: ["f0", "f1"],
        6: ["g0", "g1"],
    }
    # star: every bridge record targets cluster 1, so all triplets co-locate there
    label_two = [("b0", "x"), ("b1", "y"), ("f0", "w")]
    return clusters, label_two


def _contested_data():
    clusters = {0: ["a0", "a1", "x"], 1: ["b0", "b1"], 2: ["c0", "c1"], 3: ["d0", "d1"]}
    # one bridge record x (home 0) reached by target clusters 1 and 3
    label_two = [("b0", "x"), ("d0", "x")]
    return clusters, label_two


def _pinned_data():
    clusters = {
        0: ["a0", "a1", "x"],
        1: ["b0", "b1", "y"],
        2: ["c0", "c1"],
        3: ["d0", "d1", "z"],
        4: ["e0", "e1"],
        5: ["f0", "f1"],
        6: ["g0", "g1"],
    }
    # x (home 0) is reached by clusters 1 and 3; y and z pin both clusters to 0
    label_two = [("a0", "y"), ("a1", "z"), ("b0", "x"), ("d0", "x")]
    return clusters, label_two


def _stranded_data():
    clusters = {
        0: ["a0", "a1", "x", "c0"],
        1: ["b0", "b1", "w0"],
        2: ["c1"],
        3: ["d0", "d1"],
        4: ["e0", "e1"],
    }
    # c0 and w0 pin cluster 0 apart from clusters 1 and 3; x (home 0, targets 1
    # and 3) can then share a split with no target cluster
    label_two = [("c1", "c0"), ("d0", "w0"), ("b0", "x"), ("d1", "x")]
    return clusters, label_two


def _multi_link_data():
    clusters = {0: ["a0", "x"], 1: ["b0", "b1", "b2"], 2: ["c0", "c1"], 3: ["d0", "d1"]}
    # one bridge record with two source links and two target links
    label_two = [("b0", "x"), ("b1", "x")]
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


def _label_one(frame: pl.DataFrame):
    return [
        (
            EntityReferenceIdentifier(r["left_id"], r["left_source"]),
            EntityReferenceIdentifier(r["right_id"], r["right_source"]),
        )
        for r in frame.iter_rows(named=True)
        if r["label"] == 1
    ]


def _split_of_clusters(generator, cluster_gt) -> dict[int, str]:
    result = {}
    for name, frame in generator._splits.items():
        for ref in _frame_refs(frame):
            if ref in cluster_gt:
                result[cluster_gt[ref]] = name
    return result


@pytest.fixture
def star_data():
    return _build(*_star_data())


def test_bad_triplet_split_co_locates_all_bridge_records_in_one_split(
    star_data, caplog
):
    id_table, cluster_gt, matcher_gt = star_data
    logger = logging.getLogger("bad-triplet-test")
    generator = BadTripletSplitGenerator(split_ratio=RATIO, log=logger)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        generator.load(id_table, cluster_gt, matcher_gt).generate()

    bridges = {name: _label_two(frame) for name, frame in generator._splits.items()}
    assert sorted(map(len, bridges.values())) == [0, 0, 3]
    (bridged,) = [name for name, pairs in bridges.items() if pairs]
    split_clusters = {
        name: {cluster_gt[ref] for ref in _frame_refs(frame) if ref in cluster_gt}
        for name, frame in generator._splits.items()
    }
    for left, right in bridges[bridged]:
        for ref in (left, right):
            assert cluster_gt[ref] in split_clusters[bridged]
    assert "splits without bridges" in caplog.text
    assert "cannot be placed" not in caplog.text
    assert "rule 4 cannot be satisfied" not in caplog.text


def test_bad_triplet_split_keeps_cluster_records_disjoint(star_data):
    id_table, cluster_gt, matcher_gt = star_data
    generator = (
        BadTripletSplitGenerator(split_ratio=RATIO)
        .load(id_table, cluster_gt, matcher_gt)
        .generate()
    )

    split_records = [_frame_refs(frame) for frame in generator._splits.values()]
    assert sum(map(len, split_records)) == len(set().union(*split_records))
    assert set().union(*split_records) == set(cluster_gt)


def test_contested_bridge_record_selects_one_target_cluster(caplog):
    id_table, cluster_gt, matcher_gt = _build(*_contested_data())
    logger = logging.getLogger("contested-test")
    generator = BadTripletSplitGenerator(split_ratio=RATIO, log=logger)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        generator.load(id_table, cluster_gt, matcher_gt).generate()

    split_of = _split_of_clusters(generator, cluster_gt)
    rows = [pair for frame in generator._splits.values() for pair in _label_two(frame)]
    assert len(rows) == 1
    target, bridge_record = rows[0]
    winner = cluster_gt[target]
    loser_target = 3 if winner == 1 else 1
    assert bridge_record.label == "x"
    assert split_of[winner] == split_of[cluster_gt[bridge_record]]
    assert split_of[loser_target] != split_of[winner]
    assert generator.orphan_records == set()
    assert "rule 4 cannot be satisfied" not in caplog.text


def test_unsplittable_target_clusters_keep_all_label_two_rows(caplog):
    id_table, cluster_gt, matcher_gt = _build(*_pinned_data())
    logger = logging.getLogger("pinned-test")
    generator = BadTripletSplitGenerator(split_ratio=RATIO, log=logger)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        generator.load(id_table, cluster_gt, matcher_gt).generate()

    split_of = _split_of_clusters(generator, cluster_gt)
    assert len({split_of[cluster] for cluster in (0, 1, 3)}) == 1
    rows = {
        (left.label, right.label)
        for frame in generator._splits.values()
        for left, right in _label_two(frame)
    }
    assert rows == {("a0", "y"), ("a1", "z"), ("b0", "x"), ("d0", "x")}
    assert "rule 4 cannot be satisfied" in caplog.text


def test_unplaceable_bridge_record_is_reported_as_orphan():
    id_table, cluster_gt, matcher_gt = _build(*_stranded_data())
    generator = BadTripletSplitGenerator(
        split_ratio=RATIO, match_bridge_ratio=0, seed=SEED
    )
    generator.load(id_table, cluster_gt, matcher_gt).generate()

    assert generator.orphan_records == {EntityReferenceIdentifier("x", "source")}
    rows = [
        (left.label, right.label)
        for frame in generator._splits.values()
        for left, right in _label_two(frame)
    ]
    assert len(rows) == 2
    label_one_rows = [
        (left.label, right.label)
        for frame in generator._splits.values()
        for left, right in _label_one(frame)
    ]
    assert ("a1", "x") in label_one_rows


def test_drop_orphans_removes_unplaceable_bridge_record():
    id_table, cluster_gt, matcher_gt = _build(*_stranded_data())
    generator = BadTripletSplitGenerator(
        split_ratio=RATIO, match_bridge_ratio=0, seed=SEED, drop_orphans=True
    )
    generator.load(id_table, cluster_gt, matcher_gt).generate()

    all_refs = set().union(*(_frame_refs(f) for f in generator._splits.values()))
    assert EntityReferenceIdentifier("x", "source") not in all_refs
    rows = [pair for frame in generator._splits.values() for pair in _label_two(frame)]
    assert len(rows) == 2


def test_bridge_record_keeps_several_source_and_target_links():
    id_table, cluster_gt, matcher_gt = _build(*_multi_link_data())
    generator = (
        BadTripletSplitGenerator(split_ratio=RATIO)
        .load(id_table, cluster_gt, matcher_gt)
        .generate()
    )

    rows = [
        (left.label, right.label)
        for frame in generator._splits.values()
        for left, right in _label_two(frame)
    ]
    assert rows == [("b0", "x"), ("b1", "x")]
    label_one_rows = [
        (left.label, right.label)
        for left, right in _label_one(generator._splits["train"])
    ]
    assert ("a0", "x") in label_one_rows


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


def test_quota_counts_fill_eval_splits_when_items_are_scarce():
    generator = BadTripletSplitGenerator(split_ratio=RATIO)
    assert generator._quota_counts(0) == [0, 0, 0]
    assert generator._quota_counts(2) == [0, 1, 1]
    assert generator._quota_counts(5) == [3, 1, 1]


def test_quota_counts_fill_train_when_bridge_records_are_scarce():
    generator = BadTripletSplitGenerator(split_ratio=RATIO)
    assert generator._quota_counts(2, eval_first=False) == [2, 0, 0]
    assert generator._quota_counts(1, eval_first=False) == [1, 0, 0]
    assert generator._quota_counts(5, eval_first=False) == [3, 1, 1]


def test_bad_triplet_split_writes_configured_file_names(star_data, tmp_path):
    id_table, cluster_gt, matcher_gt = star_data
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
