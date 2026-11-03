from collections import Counter, defaultdict

import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching.evaluation.data.splits._bad_triplet_generator import (
    EXTRA_MATCH,
    NEGATIVE,
    TRIPLET_SOURCE,
    TRIPLET_TARGET,
)

FILE_NAMES = {"train": "a.csv", "dev": "b.csv", "test": "c.csv"}
SYNTHETIC_COMPONENTS = 6
SYNTHETIC_LINKS = {"train": 12, "dev": 3, "test": 3}
FODORS = "amb_fodors_zagat_nophone_a050_dir"
AFFILIATIONSTRINGS = "amb_affiliationstrings_a050_dir"
DATASETS = ["bad_triplet_synthetic", FODORS, AFFILIATIONSTRINGS]
META_COLUMNS = [
    "left_id",
    "left_source",
    "right_id",
    "right_source",
    "category",
    "bridge_id",
    "bridge_source",
]
CATEGORIES = {NEGATIVE, TRIPLET_SOURCE, EXTRA_MATCH, TRIPLET_TARGET}
GENERATED = "ambiguity-generator"


def _pair(row):
    return (
        RefId(int(row["left_id"]), row["left_source"]),
        RefId(int(row["right_id"]), row["right_source"]),
    )


def _bridge(row):
    return RefId(int(row["bridge_id"]), row["bridge_source"])


def _rows_of(meta_rows, category):
    return {
        name: [
            (_pair(row), _bridge(row)) for row in rows if row["category"] == category
        ]
        for name, rows in meta_rows.items()
    }


def test_generate_raises_when_matcher_ground_truth_has_class_three(
    make_generator, label_three_data
):
    generator = make_generator(label_three_data)

    with pytest.raises(ValueError, match="does not support classes"):
        generator.generate()


def test_get_splits_returns_one_split_per_split_name(generated, split_ratio):
    splits = generated.get_splits()

    assert list(splits) == list(split_ratio)


def test_save_writes_configured_file_names(generated, tmp_path):
    generated.save(tmp_path, file_names=FILE_NAMES)

    assert all((tmp_path / name).is_file() for name in FILE_NAMES.values())


def test_generate_logs_bridge_graph_component_count(make_generator, dataset, caplog):
    generator = make_generator(dataset)

    generator.generate()

    assert f"bridge graph: {SYNTHETIC_COMPONENTS} components" in caplog.messages


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_each_cluster_lands_in_one_split(cluster_splits):
    assert all(len(splits) == 1 for splits in cluster_splits.values())


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_bridge_footprint_lands_in_one_split(footprint_splits):
    assert all(len(splits) == 1 for splits in footprint_splits.values())


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_link_rows_keep_every_link_of_active_bridges(meta_rows, link_truth):
    for rows in meta_rows.values():
        kept = defaultdict(set)
        for row in rows:
            if row["category"] != TRIPLET_TARGET:
                continue
            kept[_bridge(row)].add(_pair(row))
        for bridge, pairs in kept.items():
            assert pairs == link_truth[bridge]


def test_link_rows_follow_split_ratio_when_components_allow(link_counts):
    assert link_counts == SYNTHETIC_LINKS


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_full_matches_follow_match_bridge_ratio(category_counts):
    for counts in category_counts.values():
        full = counts[TRIPLET_SOURCE] + counts[EXTRA_MATCH]
        assert full == 2 * counts[TRIPLET_TARGET]


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_extra_matches_do_not_involve_active_bridges(meta_rows):
    for rows in meta_rows.values():
        active = {_bridge(row) for row in rows if row["category"] == TRIPLET_TARGET}
        for row in rows:
            if row["category"] != EXTRA_MATCH:
                continue
            assert active.isdisjoint(_pair(row))


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_bridge_rows_put_the_bridge_on_the_right(meta_rows, bridges):
    for rows in meta_rows.values():
        for row in rows:
            left, right = _pair(row)
            if left in bridges or right in bridges:
                assert right in bridges
                assert left not in bridges


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_directed_labels_follow_the_ground_truth(meta_rows, split_rows):
    for name, rows in meta_rows.items():
        labels = {(left, right): label for left, right, label in split_rows[name]}
        for row in rows:
            pair = _pair(row)
            label = labels.get(pair, labels.get((pair[1], pair[0])))
            if row["category"] == TRIPLET_TARGET:
                assert label == 2
            elif row["category"] == NEGATIVE:
                assert label == 0
            else:
                assert label == 1


@pytest.mark.parametrize("dataset", ["bad_triplet_synthetic"], indirect=True)
def test_source_rows_are_capped_by_links_and_home_pairs(
    meta_rows, source_truth, link_truth
):
    for rows in meta_rows.values():
        kept = defaultdict(set)
        for row in rows:
            if row["category"] == TRIPLET_SOURCE:
                kept[_bridge(row)].add(_pair(row))
        for bridge, pairs in kept.items():
            homes = source_truth.get(bridge, frozenset())
            assert pairs <= homes
            assert len(pairs) == min(len(link_truth[bridge]), len(homes))


def test_binary_ground_truth_produces_the_directed_structure(
    make_generator, dataset, binary_dataset, select_by_source
):
    directed = make_generator(dataset).generate()
    binary = make_generator(binary_dataset, is_bridge=select_by_source).generate()

    for name in directed.split_ratio:
        directed_rows = [
            (row.pair, row.category) for row in directed._categorized[name]
        ]
        binary_rows = [(row.pair, row.category) for row in binary._categorized[name]]
        assert directed_rows == binary_rows


def test_binary_ground_truth_labels_every_positive_row_with_one(
    make_generator, binary_dataset, select_by_source
):
    binary = make_generator(binary_dataset, is_bridge=select_by_source).generate()

    for rows in binary._categorized.values():
        assert {row.label for row in rows if row.category != NEGATIVE} == {1}


def test_generate_raises_when_a_bridge_is_orphaned(make_generator, dataset):
    id_table, cluster_gt, matcher_gt = dataset
    kept = {
        pair: label
        for pair, label in matcher_gt.items()
        if label != 1 or RefId(108, GENERATED) not in pair
    }

    with pytest.raises(ValueError, match="orphan bridges"):
        make_generator((id_table, cluster_gt, kept)).generate()


def test_generate_raises_when_bridge_structure_is_malformed(make_generator, dataset):
    id_table, cluster_gt, matcher_gt = dataset
    stranger = RefId(9999, "elsewhere")
    malformed = {**matcher_gt, (stranger, RefId(108, GENERATED)): 1}

    with pytest.raises(ValueError, match="invalid bridge structure"):
        make_generator((id_table, cluster_gt, malformed)).generate()


def test_bridge_bridge_pairs_do_not_prevent_generation(generated, bridges):
    for rows in generated._categorized.values():
        for row in rows:
            assert len([ref for ref in row.pair if ref in bridges]) <= 1


def _single_split_table():
    home, bridge, target, other = (
        RefId(1, "pairs"),
        RefId(2, GENERATED),
        RefId(3, "pairs"),
        RefId(4, "pairs"),
    )
    table = InMemoryIdTable()
    for ref in (home, bridge, target, other):
        table.put(EntityReference(ref, {"id": ref.label}))
    clusters = {home: 1, bridge: 1, target: 2, other: 3}
    return table, clusters, {(home, bridge): 1, (target, bridge): 2}


def test_generate_raises_when_the_match_bridge_ratio_cannot_be_met(make_generator):
    table, clusters, matcher_gt = _single_split_table()

    with pytest.raises(ValueError, match="match-bridge ratio not met"):
        make_generator((table, clusters, matcher_gt), ratios={"train": 1}).generate()


def test_ratio_above_two_falls_back_to_the_split_pool(make_generator):
    home_1, home_2, bridge, target = (
        RefId(1, "pairs"),
        RefId(2, "pairs"),
        RefId(3, GENERATED),
        RefId(4, "pairs"),
    )
    extra_1, extra_2 = RefId(5, "pairs"), RefId(6, "pairs")
    table = InMemoryIdTable()
    for ref in (home_1, home_2, bridge, target, extra_1, extra_2):
        table.put(EntityReference(ref, {"id": ref.label}))
    clusters = {home_1: 1, home_2: 1, bridge: 1, target: 2, extra_1: 3, extra_2: 3}
    matcher_gt = {
        (home_1, home_2): 1,
        (home_1, bridge): 1,
        (target, bridge): 2,
        (extra_1, extra_2): 1,
    }

    generator = make_generator(
        (table, clusters, matcher_gt), ratios={"train": 1}, match_bridge_ratio=3
    ).generate()

    rows = generator._categorized["train"]
    counts = Counter(row.category for row in rows)
    assert counts[TRIPLET_TARGET] == 1
    assert counts[TRIPLET_SOURCE] == 1
    assert counts[EXTRA_MATCH] == 2
    pairs = {row.pair for row in rows if row.category == EXTRA_MATCH}
    assert (home_1, home_2) in pairs
    assert (extra_1, extra_2) in pairs


@pytest.mark.parametrize("dataset", [FODORS], indirect=True)
def test_ratio_falls_back_to_other_bridge_pairs_with_invalidation(generated, bridges):
    invalidated = []
    for rows in generated._categorized.values():
        active = {row.bridge for row in rows if row.category == TRIPLET_TARGET}
        for row in rows:
            if row.category != EXTRA_MATCH:
                continue
            mentioned = {ref for ref in row.pair if ref in bridges}
            if mentioned:
                assert mentioned.isdisjoint(active)
                invalidated.append(row)
    assert invalidated


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_negatives_never_pair_a_bridge_with_its_target_clusters(target_negatives):
    assert target_negatives == []


@pytest.mark.parametrize(
    "generator_options", [{"allow_target_negatives": True}], indirect=True
)
def test_allow_target_negatives_admits_bridge_target_negatives(target_negatives):
    assert target_negatives != []


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_negatives_are_never_known_pairs_in_either_orientation(negatives, dataset):
    _, _, matcher_gt = dataset

    assert [p for p in negatives if p in matcher_gt or p[::-1] in matcher_gt] == []


def test_negatives_put_bridges_on_the_right(negatives, bridges):
    for left, right in negatives:
        if left in bridges or right in bridges:
            assert right in bridges
            assert left not in bridges


def test_no_pair_appears_twice_across_splits(repeated_rows):
    assert repeated_rows == []


def test_save_writes_meta_file_with_category_columns(meta_paths):
    import polars as pl

    columns = [pl.read_csv(path).columns for path in meta_paths.values()]

    assert columns == [META_COLUMNS] * len(meta_paths)


def test_meta_file_marks_every_row_category(category_counts):
    categories = set().union(*category_counts.values())

    assert categories == CATEGORIES


def test_meta_file_has_one_row_per_split_row(meta_rows, split_rows):
    meta_lengths = {name: len(rows) for name, rows in meta_rows.items()}
    split_lengths = {name: len(rows) for name, rows in split_rows.items()}

    assert meta_lengths == split_lengths


@pytest.mark.parametrize("dataset", [FODORS], indirect=True)
def test_generate_warns_when_a_component_exceeds_the_smallest_split_target(
    make_generator, dataset, caplog
):
    make_generator(dataset).generate()

    assert "exceeds the smallest split target" in caplog.text


def test_generate_does_not_warn_when_every_component_fits_every_split(
    make_generator, dataset, caplog
):
    make_generator(dataset).generate()

    assert "exceeds the smallest split target" not in caplog.text


@pytest.mark.parametrize("dataset", [FODORS], indirect=True)
def test_negative_rows_match_the_neg_pos_ratio(generated):
    for rows in generated._categorized.values():
        positives = sum(1 for row in rows if row.label > 0)
        negatives = sum(1 for row in rows if row.label == 0)

        assert negatives == round(generated.neg_pos_ratio * positives)
