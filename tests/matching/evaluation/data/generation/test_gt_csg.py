import pytest

import numpy as np

from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from matchescu.matching.evaluation.data.generation._gt_csg import (
    GroundTruthComparisonSpaceGenerator,
)
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId


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
        save_comparisons=False,
        save_clusters=False,
    )


@pytest.fixture
def multiclass_matcher_gt(ref_ids):
    gt = {}
    for i in range(7):
        gt[(ref_ids[i], ref_ids[i + 10])] = 1
    for i in range(7, 10):
        gt[(ref_ids[i], ref_ids[i + 10])] = 2
    return gt


@pytest.fixture
def csg_multiclass(id_table, multiclass_matcher_gt, cluster_gt):
    return GroundTruthComparisonSpaceGenerator(
        id_table,
        multiclass_matcher_gt,
        cluster_gt=cluster_gt,
        neg_pos_ratio=3.0,
        match_bridge_ratio=2.0,
        seed=123,
        save_comparisons=False,
        save_clusters=False,
    )


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
        save_comparisons=False,
        save_clusters=False,
    )
    assert csg.true_clusters is cluster_gt
    assert csg.true_matches is undirected_ground_truth


def test_init_stores_default_ratios(id_table, undirected_ground_truth, cluster_gt):
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        save_comparisons=False,
        save_clusters=False,
    )
    assert csg.neg_pos_ratio == 5.0
    assert csg.match_bridge_ratio == 2.0
    assert csg.max_total_samples is None
    assert csg.random_state == 42


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
            save_comparisons=False,
            save_clusters=False,
        )


@pytest.mark.parametrize(
    "target, available, expected",
    [
        (4, 5, 4),  # target < available -> target
        (5, 4, 4),  # target > available  -> available
        (1, 4, 3),  # target < MIN_PER_CLASS -> MIN_PER_CLASS
        (1, 2, 3),  # both < MIN_PER_CLASS -> MIN_PER_CLASS
        (0, 0, 3),  # edge: both zero → MIN_PER_CLASS
        (3, 3, 3),  # exact MIN_PER_CLASS
    ],
)
def test_clamp(target, available, expected):
    actual = GroundTruthComparisonSpaceGenerator._clamp(target, available)

    assert actual == expected


# ──────────────────────────────────────────────────────────────────────
# Tests — _apply_cap
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
    # Every class gets at least MIN_PER_CLASS
    for v in result.values():
        assert v >= GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS


def test_apply_cap_preserves_proportions_roughly(csg_binary):
    csg_binary.max_total_samples = 100
    targets = {0: 500, 1: 100, 2: 50}
    result = csg_binary._apply_cap(targets)
    assert sum(result.values()) == 100
    # class 0 should still be the largest
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_apply_cap_raises_when_budget_too_small(csg_binary):
    csg_binary.max_total_samples = 2  # less than MIN_PER_CLASS * n_classes
    targets = {0: 100, 1: 50}
    with pytest.raises(ValueError, match="too small"):
        csg_binary._apply_cap(targets)


def test_apply_cap_exact_minimum(csg_binary):
    """Budget == MIN_PER_CLASS * n_classes → every class gets exactly MIN."""
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
    # r0-r1 appears many times (high freq), r2-r3 and r4-r5 appear once
    pool = [
        (rids[0], rids[1]),
        (rids[0], rids[1]),  # duplicate bumps frequency
        (rids[0], rids[1]),
        (rids[2], rids[3]),
        (rids[4], rids[5]),
    ]
    result = csg_binary._select_diverse(pool, n=2)
    # The two low-frequency pairs should be selected
    result_set = {tuple(p) for p in result}
    assert (rids[2], rids[3]) in result_set or (rids[4], rids[5]) in result_set


def test_select_diverse_is_deterministic(csg_binary, ref_ids):
    pool = [(ref_ids[i], ref_ids[i + 10]) for i in range(10)]
    r1 = csg_binary._select_diverse(pool, n=5)
    # Reset RNG to same state
    csg_binary._rng = np.random.RandomState(42)
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
    gt = {}
    csg = GroundTruthComparisonSpaceGenerator(
        tiny_table,
        gt,
        cluster_gt=frozenset(),
        seed=0,
        save_comparisons=False,
        save_clusters=False,
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
    cs = csg_binary()

    pairs = set(cs)

    has_pos = any(p in undirected_ground_truth for p in pairs)
    has_neg = any(p not in undirected_ground_truth for p in pairs)
    assert has_pos, "comparison space should contain positive pairs"
    assert has_neg, "comparison space should contain negative pairs"


def test_call_multiclass(csg_multiclass, multiclass_matcher_gt):
    cs = csg_multiclass()
    pairs = list(cs)

    assert len(pairs) > 0
    labels_seen = set()
    for p in pairs:
        labels_seen.add(multiclass_matcher_gt.get(p, 0))

    assert 0 in labels_seen
    assert 1 in labels_seen
    assert 2 in labels_seen


def test_call_with_max_total_samples(id_table, undirected_ground_truth, cluster_gt):
    cap = 25
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=5.0,
        max_total_samples=cap,
        seed=42,
        save_comparisons=False,
        save_clusters=False,
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
        save_comparisons=True,
        save_clusters=False,
    )
    csg(output_path=out)
    assert out.exists()

    import polars as pl

    df = pl.read_csv(out)
    assert "left_id" in df.columns
    assert "label" in df.columns
    counts = {row[0]: row[1] for row in df["label"].value_counts(sort=True).iter_rows()}
    assert counts[0] / counts[1] == 2.0
    assert len(df) > 0


def test_write_clusters_csv(id_table, undirected_ground_truth, cluster_gt, tmp_path):
    clusters_out = tmp_path / "clusters.csv"
    csg = GroundTruthComparisonSpaceGenerator(
        id_table,
        undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=2.0,
        seed=42,
        save_comparisons=False,
        save_clusters=True,
    )
    csg(clusters_output_path=clusters_out)

    assert clusters_out.exists()
    import polars as pl

    df = pl.read_csv(clusters_out)
    assert "cluster_id" in df.columns
    assert "id" in df.columns
    assert "source" in df.columns


def test_save_disabled_returns_empty_df(csg_binary):
    from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
    import polars as pl

    cs = InMemoryComparisonSpace()
    df_cmp = csg_binary._write_comparisons_csv(cs, None)
    df_cls = csg_binary._write_clusters_csv(cs, None)
    assert isinstance(df_cmp, pl.DataFrame) and len(df_cmp) == 0
    assert isinstance(df_cls, pl.DataFrame) and len(df_cls) == 0


def test_true_clusters_property(csg_binary, cluster_gt):
    assert csg_binary.true_clusters == cluster_gt


def test_true_matches_property(csg_binary, undirected_ground_truth):
    assert csg_binary.true_matches is undirected_ground_truth


def test_reproducibility_with_same_seed(id_table, undirected_ground_truth, cluster_gt):
    """Two generators with the same seed should produce identical comparison spaces."""
    kwargs = dict(
        id_table=id_table,
        matcher_gt=undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=2.0,
        seed=99,
        save_comparisons=False,
        save_clusters=False,
    )
    cs1 = list(GroundTruthComparisonSpaceGenerator(**kwargs)())
    cs2 = list(GroundTruthComparisonSpaceGenerator(**kwargs)())
    assert cs1 == cs2


def test_different_seeds_produce_different_negatives(
    id_table, undirected_ground_truth, cluster_gt
):
    kwargs = dict(
        id_table=id_table,
        matcher_gt=undirected_ground_truth,
        cluster_gt=cluster_gt,
        neg_pos_ratio=2.0,
        save_comparisons=False,
        save_clusters=False,
    )
    cs1 = set(GroundTruthComparisonSpaceGenerator(seed=1, **kwargs)())
    cs2 = set(GroundTruthComparisonSpaceGenerator(seed=999, **kwargs)())
    # The positive pairs overlap, but negatives should differ
    assert cs1 != cs2
