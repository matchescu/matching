from collections import Counter
from typing import NamedTuple

import polars as pl
import pytest
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from matchescu.matching.evaluation.data.generation._gt_csg import (
    GroundTruthComparisonSpaceGenerator,
)

BRIDGE_SOURCE = "ambiguity-generator"


@pytest.fixture
def benchmark_data(affils_dir, affils_traits, affils_id_col):
    return (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters("affiliationstrings_clusters.csv")
    )


class _StubRef:
    """Mimics an entity reference with an `.id` attribute."""

    def __init__(self, ref_id: RefId):
        self.id = ref_id


class _StubIdTable(IdTable):
    """Minimal stand-in for IdTable."""

    def __init__(self, ref_ids: list[RefId]):
        self._refs = [_StubRef(r) for r in ref_ids]
        self._ids = ref_ids

    def ids(self):
        return self._ids

    def __iter__(self):
        return iter(self._refs)

    def __len__(self):
        return len(self._refs)

    def __contains__(self, item):
        return item in self._ids


def _make_ref_ids(n: int, source: str = "src") -> list[RefId]:
    return [RefId(label=f"r{i}", source=source) for i in range(n)]


@pytest.fixture
def ref_ids():
    """20 fake reference IDs spread across two sources."""
    return _make_ref_ids(10, "A") + _make_ref_ids(10, "B")


@pytest.fixture
def id_table(ref_ids):
    return _StubIdTable(ref_ids)


@pytest.fixture
def undirected_ground_truth(ref_ids):
    """Simple binary (class-1 only) matcher ground truth with 10 positive pairs."""
    gt = {}
    for i in range(10):
        gt[(ref_ids[i], ref_ids[i + 10])] = 1
    return gt


@pytest.fixture
def cluster_gt(ref_ids) -> frozenset[frozenset[RefId]]:
    """10 clusters of size 2 (matching the binary_matcher_gt)."""
    return frozenset(frozenset({ref_ids[i], ref_ids[i + 10]}) for i in range(10))


@pytest.fixture
def csg_binary(id_table, undirected_ground_truth, cluster_gt):
    """CSG with binary GT, saving disabled so __call__ doesn't need paths."""
    return GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=2.0,
        seed=42,
        save=False,
    )


class BridgedRefs(NamedTuple):
    home_1: RefId
    home_2: RefId
    bridge: RefId
    target: RefId
    other_1: RefId
    other_2: RefId


@pytest.fixture
def bridged_refs():
    return BridgedRefs(
        home_1=RefId("h1", "data"),
        home_2=RefId("h2", "data"),
        bridge=RefId("b1", BRIDGE_SOURCE),
        target=RefId("t1", "data"),
        other_1=RefId("o1", "data"),
        other_2=RefId("o2", "data"),
    )


@pytest.fixture
def bridged_id_table(bridged_refs):
    return _StubIdTable(list(bridged_refs))


@pytest.fixture
def bridged_cluster_gt(bridged_refs):
    r = bridged_refs
    return frozenset(
        {
            frozenset({r.home_1, r.home_2, r.bridge}),
            frozenset({r.target}),
            frozenset({r.other_1, r.other_2}),
        }
    )


@pytest.fixture
def directed_bridged_gt(bridged_refs):
    r = bridged_refs
    return {
        (r.home_1, r.home_2): 1,
        (r.home_1, r.bridge): 1,
        (r.home_2, r.bridge): 1,
        (r.target, r.bridge): 2,
        (r.other_1, r.other_2): 1,
    }


@pytest.fixture
def binary_bridged_gt(directed_bridged_gt):
    return {
        pair: 1 if label == 2 else label for pair, label in directed_bridged_gt.items()
    }


def _bridged_csg(id_table, matcher_gt, cluster_gt, **kwargs):
    options = {
        "neg_pos_ratio": 2.0,
        "match_bridge_ratio": 2.0,
        "seed": 42,
        "save": False,
    }
    options.update(kwargs)
    return GroundTruthComparisonSpaceGenerator(
        id_table, matcher_gt, cluster_gt=cluster_gt, **options
    )


@pytest.fixture
def csg_bridged(bridged_id_table, directed_bridged_gt, bridged_cluster_gt):
    return _bridged_csg(bridged_id_table, directed_bridged_gt, bridged_cluster_gt)


@pytest.fixture
def csg_bridged_binary(bridged_id_table, binary_bridged_gt, bridged_cluster_gt):
    return _bridged_csg(
        bridged_id_table,
        binary_bridged_gt,
        bridged_cluster_gt,
        is_bridge=lambda ref: ref.source == BRIDGE_SOURCE,
    )


def _labels(csg, pairs):
    return Counter(csg._label_of(pair) for pair in pairs)


# ──────────────────────────────────────────────────────────────────────
# Tests — initialization
# ──────────────────────────────────────────────────────────────────────


def test_init_without_cluster_gt(benchmark_data):
    """Integration: cluster GT is inferred via EquivalenceClassPartitioner."""
    csg = GroundTruthComparisonSpaceGenerator(
        benchmark_data.id_table, benchmark_data.true_matches, None
    )
    assert len(csg.true_clusters) == 330


def test_init_with_explicit_cluster_gt(id_table, undirected_ground_truth, cluster_gt):
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        seed=0,
        save=False,
    )
    assert csg.true_clusters is cluster_gt
    assert csg.true_matches is undirected_ground_truth


def test_init_stores_default_ratios(id_table, undirected_ground_truth, cluster_gt):
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        save=False,
    )
    assert csg.neg_pos_ratio == 5.0
    assert csg.match_bridge_ratio == 2.0
    assert csg.max_total_samples is None
    assert csg.is_bridge is None
    assert csg.allow_target_negatives is False


def test_init_ref_id_cluster_map_populated(csg_binary, ref_ids):
    for rid in ref_ids:
        assert rid in csg_binary._ref_id_cluster_map


def test_init_raises_on_multiclass_without_cluster_gt(id_table):
    rids = list(id_table.ids())
    gt = {
        (rids[0], rids[10]): 1,
        (rids[1], rids[11]): 2,
        (rids[2], rids[12]): 3,
    }
    with pytest.raises(
        ValueError, match="can't perform equivalence class partitioning"
    ):
        GroundTruthComparisonSpaceGenerator(
            id_table,
            gt,
            cluster_gt=None,
            save=False,
        )


@pytest.mark.parametrize(
    "target, available, expected",
    [
        (4, 5, 4),
        (5, 4, 4),
        (1, 4, 3),
        (1, 2, 3),
        (0, 0, 3),
        (3, 3, 3),
    ],
)
def test_clamp(target, available, expected):
    actual = GroundTruthComparisonSpaceGenerator._clamp(target, available)

    assert actual == expected


# ──────────────────────────────────────────────────────────────────────
# Tests — legacy (bridge-less) sampling
# ──────────────────────────────────────────────────────────────────────


def test_apply_cap_no_limit(csg_binary):
    targets = {0: 100, 1: 20}
    result = csg_binary._apply_cap(targets)
    assert result == targets


def test_apply_cap_under_budget(csg_binary):
    csg_binary.max_total_samples = 500
    targets = {0: 100, 1: 20}
    result = csg_binary._apply_cap(targets)
    assert result == targets


def test_apply_cap_scales_down(csg_binary):
    csg_binary.max_total_samples = 50
    targets = {0: 200, 1: 50}
    result = csg_binary._apply_cap(targets)
    assert sum(result.values()) == 50
    for v in result.values():
        assert v >= GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS


def test_apply_cap_preserves_proportions_roughly(csg_binary):
    csg_binary.max_total_samples = 100
    targets = {0: 500, 1: 100, 2: 50}
    result = csg_binary._apply_cap(targets)
    assert sum(result.values()) == 100
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_apply_cap_raises_when_budget_too_small(csg_binary):
    csg_binary.max_total_samples = 2
    targets = {0: 100, 1: 50}
    with pytest.raises(ValueError, match="too small"):
        csg_binary._apply_cap(targets)


def test_apply_cap_exact_minimum(csg_binary):
    n_classes = 3
    csg_binary.max_total_samples = (
        GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS * n_classes
    )
    targets = {0: 100, 1: 50, 2: 30}
    result = csg_binary._apply_cap(targets)
    assert sum(result.values()) == csg_binary.max_total_samples
    for v in result.values():
        assert v >= GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS


def test_select_diverse_returns_all_when_n_larger_than_pool_size(csg_binary, ref_ids):
    pool = [(ref_ids[0], ref_ids[10]), (ref_ids[1], ref_ids[11])]
    result = csg_binary._select_diverse(pool, n=10)
    assert len(result) == len(pool)


def test_select_diverse_returns_n_items(csg_binary, ref_ids):
    pool = [(ref_ids[i], ref_ids[i + 10]) for i in range(10)]
    result = csg_binary._select_diverse(pool, n=4)
    assert len(result) == 4


def test_select_diverse_prefers_low_frequency_pairs(csg_binary):
    rids = _make_ref_ids(6, "X")
    pool = [
        (rids[0], rids[1]),
        (rids[0], rids[1]),
        (rids[0], rids[1]),
        (rids[2], rids[3]),
        (rids[4], rids[5]),
    ]
    result = csg_binary._select_diverse(pool, n=2)
    result_set = {tuple(p) for p in result}
    assert (rids[2], rids[3]) in result_set or (rids[4], rids[5]) in result_set


def test_select_diverse_is_deterministic(csg_binary, ref_ids):
    pool = [(ref_ids[i], ref_ids[i + 10]) for i in range(10)]
    r1 = csg_binary._select_diverse(pool, n=5)
    csg_binary.seed = 42
    r2 = csg_binary._select_diverse(pool, n=5)
    assert r1 == r2


def test_generate_negatives_count(csg_binary, undirected_ground_truth):
    negs = csg_binary._generate_negatives(15, set(undirected_ground_truth))
    assert len(negs) == 15


def test_generate_negatives_no_self_pairs(csg_binary, undirected_ground_truth):
    negs = csg_binary._generate_negatives(20, set(undirected_ground_truth))
    for left, right in negs:
        assert left != right


def test_generate_negatives_excludes_positives(csg_binary, undirected_ground_truth):
    exclude = set(undirected_ground_truth)
    negs = csg_binary._generate_negatives(20, exclude)
    for pair in negs:
        assert pair not in exclude
        assert (pair[1], pair[0]) not in exclude


def test_generate_negatives_no_same_cluster(csg_binary, undirected_ground_truth):
    negs = csg_binary._generate_negatives(20, set(undirected_ground_truth))
    cmap = csg_binary._ref_id_cluster_map
    for left, right in negs:
        cl_l = cmap.get(left)
        cl_r = cmap.get(right)
        if cl_l is not None and cl_r is not None:
            assert cl_l != cl_r


def test_generate_negatives_no_duplicates(csg_binary, undirected_ground_truth):
    negs = csg_binary._generate_negatives(20, set(undirected_ground_truth))
    fwd_set = set()
    for left, right in negs:
        assert (left, right) not in fwd_set
        assert (right, left) not in fwd_set
        fwd_set.add((left, right))


def test_generate_negatives_raises_on_tiny_pool(cluster_gt):
    single_ref = _make_ref_ids(1, "Z")
    tiny_table = _StubIdTable(single_ref)
    csg = GroundTruthComparisonSpaceGenerator(
        tiny_table,
        {},
        cluster_gt=frozenset(),
        seed=0,
        save=False,
    )
    with pytest.raises(ValueError, match="< 2 entries"):
        csg._generate_negatives(5, set())


def test_call_returns_comparison_space(csg_binary):
    cs = csg_binary()
    pairs = list(cs)

    assert len(pairs) > 0
    for pair in pairs:
        assert len(pair) == 2


def test_call_binary_contains_positives_and_negatives(
    csg_binary, undirected_ground_truth
):
    pairs = set(csg_binary())

    assert any(p in undirected_ground_truth for p in pairs)
    assert any(p not in undirected_ground_truth for p in pairs)


def test_legacy_path_filters_excluded_positives(
    id_table, undirected_ground_truth, cluster_gt, ref_ids
):
    excluded = [(ref_ids[0], ref_ids[10])]
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        excluded=excluded,
        neg_pos_ratio=2.0,
        seed=42,
        save=False,
    )

    pairs = list(csg())

    assert (ref_ids[0], ref_ids[10]) not in pairs
    assert (ref_ids[10], ref_ids[0]) not in pairs
    assert _labels(csg, pairs) == {1: 9, 0: 18}


def test_call_with_max_total_samples(id_table, undirected_ground_truth, cluster_gt):
    cap = 25
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=5.0,
        max_total_samples=cap,
        seed=42,
        save=False,
    )

    cs = csg()

    assert len(list(cs)) <= cap


def test_write_comparisons_csv(id_table, undirected_ground_truth, cluster_gt, tmp_path):
    out = tmp_path / "comparisons.csv"
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=2.0,
        seed=42,
        save=True,
    )
    csg(output_path=out)
    assert out.exists()

    df = pl.read_csv(out)
    assert "left_id" in df.columns
    assert "label" in df.columns
    counts = {row[0]: row[1] for row in df["label"].value_counts(sort=True).iter_rows()}
    assert counts[0] / counts[1] == 2.0
    assert len(df) > 0


def test_save_disabled_returns_empty_df(csg_binary):
    from matchescu.reference_store.comparison_space import InMemoryComparisonSpace

    cs = InMemoryComparisonSpace()
    df_cmp = csg_binary._write_comparisons_csv(cs, None)
    assert isinstance(df_cmp, pl.DataFrame) and len(df_cmp) == 0


def test_true_clusters_property(csg_binary, cluster_gt):
    assert csg_binary.true_clusters == cluster_gt


def test_true_matches_property(csg_binary, undirected_ground_truth):
    assert csg_binary.true_matches is undirected_ground_truth


def test_reproducibility_with_same_seed(id_table, undirected_ground_truth, cluster_gt):
    kwargs = {
        "id_table": id_table,
        "matcher_gt": undirected_ground_truth,
        "cluster_gt": cluster_gt,
        "neg_pos_ratio": 2.0,
        "seed": 99,
        "save": False,
    }
    cs1 = list(GroundTruthComparisonSpaceGenerator(**kwargs)())
    cs2 = list(GroundTruthComparisonSpaceGenerator(**kwargs)())
    assert cs1 == cs2


def test_different_seeds_produce_different_negatives(
    id_table, undirected_ground_truth, cluster_gt
):
    kwargs = {
        "id_table": id_table,
        "matcher_gt": undirected_ground_truth,
        "cluster_gt": cluster_gt,
        "neg_pos_ratio": 2.0,
        "save": False,
    }
    cs1 = set(GroundTruthComparisonSpaceGenerator(seed=1, **kwargs)())
    cs2 = set(GroundTruthComparisonSpaceGenerator(seed=999, **kwargs)())
    assert cs1 != cs2


def test_csv_matches_returned_order_and_is_reproducible(tmp_path):
    refs = _make_ref_ids(20, "S")
    id_table = _StubIdTable(refs)
    gt = {(refs[i], refs[i + 10]): 1 for i in range(10)}
    cluster_gt = frozenset(frozenset({refs[i], refs[i + 10]}) for i in range(10))
    kwargs = {
        "id_table": id_table,
        "matcher_gt": gt,
        "cluster_gt": cluster_gt,
        "neg_pos_ratio": 2.0,
        "seed": 77,
        "save": True,
    }

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_cs = list(
        GroundTruthComparisonSpaceGenerator(**kwargs)(output_path=first_path)
    )
    second_cs = list(
        GroundTruthComparisonSpaceGenerator(**kwargs)(output_path=second_path)
    )

    assert first_cs == second_cs
    assert first_path.read_bytes() == second_path.read_bytes()

    csv_rows = list(pl.read_csv(first_path).iter_rows(named=True))
    csv_pairs = [
        (
            row["left_id"],
            row["left_source"],
            row["right_id"],
            row["right_source"],
        )
        for row in csv_rows
    ]
    returned_pairs = [
        (left.label, left.source, right.label, right.source) for left, right in first_cs
    ]
    assert csv_pairs == returned_pairs


# ──────────────────────────────────────────────────────────────────────
# Tests — bridge-aware sampling
# ──────────────────────────────────────────────────────────────────────


def test_bridged_space_respects_the_match_bridge_ratio(csg_bridged):
    pairs = list(csg_bridged())

    assert _labels(csg_bridged, pairs) == {2: 1, 1: 2, 0: 6}


def test_bridged_space_puts_the_bridge_on_the_right(csg_bridged, bridged_refs):
    bridges = {bridged_refs.bridge}

    for left, right in csg_bridged():
        if left in bridges or right in bridges:
            assert right in bridges
            assert left not in bridges


def test_binary_bridged_space_matches_the_directed_space(
    csg_bridged, csg_bridged_binary
):
    assert list(csg_bridged()) == list(csg_bridged_binary())


def test_binary_bridged_space_labels_every_positive_row_with_one(csg_bridged_binary):
    pairs = list(csg_bridged_binary())

    assert _labels(csg_bridged_binary, pairs) == {1: 3, 0: 6}


def test_bridged_reproducibility(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt
):
    first = list(
        _bridged_csg(bridged_id_table, directed_bridged_gt, bridged_cluster_gt)()
    )
    second = list(
        _bridged_csg(bridged_id_table, directed_bridged_gt, bridged_cluster_gt)()
    )

    assert first == second


def test_hard_excluded_pair_is_never_sampled(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt, bridged_refs
):
    r = bridged_refs
    excluded = [(r.other_1, r.other_2), (r.home_1, r.home_2)]
    csg = _bridged_csg(
        bridged_id_table,
        directed_bridged_gt,
        bridged_cluster_gt,
        excluded=excluded,
    )

    pairs = list(csg())

    assert (r.other_1, r.other_2) not in pairs
    assert (r.other_2, r.other_1) not in pairs
    assert (r.home_1, r.home_2) not in pairs
    assert (r.home_2, r.home_1) not in pairs
    assert _labels(csg, pairs) == {2: 1, 1: 2, 0: 6}


def test_avoided_bridge_is_bent_when_links_are_needed(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt, bridged_refs
):
    r = bridged_refs
    excluded = [(r.home_1, r.bridge), (r.home_2, r.bridge)]
    csg = _bridged_csg(
        bridged_id_table,
        directed_bridged_gt,
        bridged_cluster_gt,
        excluded=excluded,
    )

    pairs = list(csg())

    assert (r.target, r.bridge) in pairs
    assert _labels(csg, pairs) == {2: 1, 1: 2, 0: 6}


def test_demoted_bridge_hard_bans_the_exact_pair(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt, bridged_refs
):
    r = bridged_refs
    csg = _bridged_csg(
        bridged_id_table,
        directed_bridged_gt,
        bridged_cluster_gt,
        excluded=[(r.home_1, r.bridge)],
    )

    pairs = list(csg())

    assert (r.home_1, r.bridge) not in pairs
    assert (r.bridge, r.home_1) not in pairs
    assert (r.target, r.bridge) in pairs
    assert (r.home_2, r.bridge) in pairs


def test_bridged_cap_is_an_upper_bound():
    home_1, home_2, bridge_1, target_1 = (
        RefId("h1", "data"),
        RefId("h2", "data"),
        RefId("b1", BRIDGE_SOURCE),
        RefId("t1", "data"),
    )
    home_3, home_4, bridge_2, target_2 = (
        RefId("h3", "data"),
        RefId("h4", "data"),
        RefId("b2", BRIDGE_SOURCE),
        RefId("t2", "data"),
    )
    table = _StubIdTable(
        [home_1, home_2, bridge_1, target_1, home_3, home_4, bridge_2, target_2]
    )
    clusters = frozenset(
        {
            frozenset({home_1, home_2, bridge_1}),
            frozenset({target_1}),
            frozenset({home_3, home_4, bridge_2}),
            frozenset({target_2}),
        }
    )
    gt = {
        (home_1, home_2): 1,
        (home_1, bridge_1): 1,
        (home_2, bridge_1): 1,
        (target_1, bridge_1): 2,
        (home_3, home_4): 1,
        (home_3, bridge_2): 1,
        (home_4, bridge_2): 1,
        (target_2, bridge_2): 2,
    }
    csg = GroundTruthComparisonSpaceGenerator(
        table,
        gt,
        cluster_gt=clusters,
        neg_pos_ratio=2.0,
        match_bridge_ratio=2.0,
        max_total_samples=9,
        seed=5,
        save=False,
    )

    pairs = list(csg())

    assert len(pairs) <= 9
    assert _labels(csg, pairs) == {2: 1, 1: 2, 0: 6}


def test_bridged_cap_too_small_raises(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt
):
    csg = _bridged_csg(
        bridged_id_table,
        directed_bridged_gt,
        bridged_cluster_gt,
        max_total_samples=8,
    )

    with pytest.raises(ValueError, match="too small"):
        csg()


def test_bridged_space_raises_on_orphan_bridge(
    bridged_id_table, bridged_cluster_gt, bridged_refs
):
    r = bridged_refs
    gt = {
        (r.home_1, r.home_2): 1,
        (r.target, r.bridge): 2,
        (r.other_1, r.other_2): 1,
    }
    csg = _bridged_csg(bridged_id_table, gt, bridged_cluster_gt)

    with pytest.raises(ValueError, match="orphan bridges"):
        csg()


def test_bridged_space_raises_on_malformed_bridge_structure(
    bridged_id_table, directed_bridged_gt, bridged_cluster_gt, bridged_refs
):
    stranger = RefId("stranger", "elsewhere")
    gt = {**directed_bridged_gt, (stranger, bridged_refs.bridge): 1}
    csg = _bridged_csg(bridged_id_table, gt, bridged_cluster_gt)

    with pytest.raises(ValueError, match="invalid bridge structure"):
        csg()


def test_csv_resolves_flipped_home_pair_labels(tmp_path, bridged_refs):
    r = bridged_refs
    table = _StubIdTable(list(r))
    clusters = frozenset(
        {
            frozenset({r.home_1, r.home_2, r.bridge}),
            frozenset({r.target}),
            frozenset({r.other_1, r.other_2}),
        }
    )
    gt = {
        (r.home_1, r.home_2): 1,
        (r.bridge, r.home_1): 1,
        (r.bridge, r.home_2): 1,
        (r.target, r.bridge): 2,
        (r.other_1, r.other_2): 1,
    }
    csg = _bridged_csg(table, gt, clusters, save=True)
    out = tmp_path / "comparison.csv"

    csg(output_path=out)

    df = pl.read_csv(out)
    counts = {row[0]: row[1] for row in df["label"].value_counts(sort=True).iter_rows()}
    assert counts == {0: 6, 1: 2, 2: 1}
