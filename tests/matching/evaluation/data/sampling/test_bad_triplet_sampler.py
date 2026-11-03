import polars as pl
import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching.evaluation.data.sampling import BadTriplet, BadTripletSampler

GENERATED = "ambiguity-generator"
HOME = "home"
TARGET = "target"
OTHER = "other"
DATASETS = ["bad_triplet_synthetic", "amb_affiliationstrings_a050_dir"]

H1, H2 = RefId(1, HOME), RefId(2, HOME)
BRIDGE = RefId(100, GENERATED)
T1, T2 = RefId(3, TARGET), RefId(4, TARGET)
U1, U2 = RefId(5, OTHER), RefId(6, OTHER)
REFS = (H1, H2, BRIDGE, T1, T2, U1, U2)
CLUSTERS = {H1: 1, H2: 1, BRIDGE: 1, T1: 2, T2: 3, U1: 4, U2: 4}
DIRECTED_GT = {
    (H1, BRIDGE): 1,
    (H2, BRIDGE): 1,
    (T1, BRIDGE): 2,
    (T2, BRIDGE): 2,
    (U1, U2): 1,
}
BINARY_GT = {pair: 1 if label == 2 else label for pair, label in DIRECTED_GT.items()}


def by_source(ref):
    return ref.source == GENERATED


def by_label_two(gt):
    bridges = {right for (_, right), label in gt.items() if label == 2}
    return lambda ref: ref in bridges


def build_sampler(gt=None, clusters=None, refs=None, is_bridge=None):
    gt = DIRECTED_GT if gt is None else gt
    clusters = CLUSTERS if clusters is None else clusters
    refs = REFS if refs is None else refs
    table = InMemoryIdTable()
    for ref in refs:
        table.put(EntityReference(ref, {"id": ref.label}))
    predicate = by_source if is_bridge is None else is_bridge
    return BadTripletSampler(table, dict(gt), dict(clusters), predicate)


def _rows(path):
    return list(pl.read_csv(path, infer_schema_length=0).iter_rows(named=True))


def load_dataset(directory):
    name = directory.name
    table = InMemoryIdTable()
    for row in _rows(directory / f"{name}_ids.csv"):
        ref = RefId(int(row["id"]), row["source"])
        table.put(EntityReference(ref, {"id": ref.label}))
    clusters = {
        RefId(int(row["id"]), row["source"]): int(row["cluster_id"])
        for row in _rows(directory / f"{name}_cluster_mapping.csv")
    }
    gt = {
        (
            RefId(int(row["left_id"]), row["left_source"]),
            RefId(int(row["right_id"]), row["right_source"]),
        ): int(row["label"])
        for row in _rows(directory / f"{name}_mapping.csv")
    }
    return table, clusters, gt


def declared(gt, pair):
    return bool(gt.get(pair) or gt.get((pair[1], pair[0])))


def test_index_separates_home_pairs_from_links():
    index = build_sampler().index()

    assert index.home_pairs == {BRIDGE: ((H1, BRIDGE), (H2, BRIDGE))}
    assert index.link_pairs == {BRIDGE: ((T1, BRIDGE), (T2, BRIDGE))}


def test_index_derives_bridges_from_the_predicate():
    assert build_sampler().index().bridges == frozenset({BRIDGE})


def test_index_reports_bridge_bridge_pair_as_issue():
    bridge_2 = RefId(101, GENERATED)
    gt = {**DIRECTED_GT, (BRIDGE, bridge_2): 1}
    clusters = {**CLUSTERS, bridge_2: 1}

    index = build_sampler(gt=gt, refs=REFS + (bridge_2,), clusters=clusters).index()

    assert ((BRIDGE, bridge_2), "bridge_bridge") in index.issues


def test_index_reports_pair_with_unknown_record_as_issue():
    stranger = RefId(999, OTHER)
    gt = {**DIRECTED_GT, (stranger, BRIDGE): 2}

    index = build_sampler(gt=gt).index()

    assert ((stranger, BRIDGE), "unknown_ref") in index.issues


def test_index_reports_unclustered_bridge_pair_as_issue():
    clusters = {ref: cluster for ref, cluster in CLUSTERS.items() if ref != BRIDGE}

    index = build_sampler(clusters=clusters).index()

    assert ((H1, BRIDGE), "unclustered") in index.issues
    assert index.link_pairs == {}


def test_index_reports_bridge_without_links_as_orphan():
    index = build_sampler(gt={(H1, BRIDGE): 1}).index()

    assert index.orphan_bridges == ((BRIDGE, "no_links"),)


def test_index_reports_bridge_without_home_pairs_as_orphan():
    index = build_sampler(gt={(T1, BRIDGE): 2}).index()

    assert index.orphan_bridges == ((BRIDGE, "no_home_pairs"),)


def test_index_preserves_ground_truth_order():
    reordered_gt = {
        (H2, BRIDGE): 1,
        (H1, BRIDGE): 1,
        (T2, BRIDGE): 2,
        (T1, BRIDGE): 2,
    }

    index = build_sampler(gt=reordered_gt).index()

    assert index.home_pairs == {BRIDGE: ((H2, BRIDGE), (H1, BRIDGE))}
    assert index.link_pairs == {BRIDGE: ((T2, BRIDGE), (T1, BRIDGE))}


def test_sample_combines_every_link_with_every_home():
    sample = build_sampler().sample()

    assert len(sample.triplets) == 4
    assert {(triplet.target, triplet.home) for triplet in sample.triplets} == {
        (T1, H1),
        (T1, H2),
        (T2, H1),
        (T2, H2),
    }


def test_sample_follows_ground_truth_order():
    sample = build_sampler().sample()

    assert sample.triplets == (
        BadTriplet(H1, BRIDGE, T1),
        BadTriplet(H2, BRIDGE, T1),
        BadTriplet(H1, BRIDGE, T2),
        BadTriplet(H2, BRIDGE, T2),
    )


def test_sample_keeps_triplets_structurally_sound():
    sample = build_sampler().sample()

    for triplet in sample.triplets:
        assert CLUSTERS[triplet.home] == CLUSTERS[triplet.bridge]
        assert CLUSTERS[triplet.target] != CLUSTERS[triplet.bridge]
        assert declared(DIRECTED_GT, (triplet.home, triplet.bridge))
        assert declared(DIRECTED_GT, (triplet.target, triplet.bridge))


def test_sample_is_deterministic():
    first = build_sampler().sample()
    second = build_sampler().sample()

    assert first.triplets == second.triplets


def test_sample_preserves_recorded_pair_orientation():
    gt = {(BRIDGE, H1): 1, (H2, BRIDGE): 1, (T1, BRIDGE): 2, (T2, BRIDGE): 2}

    sample = build_sampler(gt=gt).sample()

    assert {triplet.home for triplet in sample.triplets} <= {H1, H2}
    assert {triplet.target for triplet in sample.triplets} == {T1, T2}


def test_exclusion_of_pair_without_bridge_becomes_hard_ban():
    baseline = build_sampler().sample()

    sample = build_sampler().sample(exclusions=[(U1, U2)])

    assert sample.exclusions.hard_exclusions == frozenset({frozenset({U1, U2})})
    assert sample.triplets == baseline.triplets


def test_exclusion_of_declared_negative_pair_becomes_hard_ban():
    gt = {**DIRECTED_GT, (T1, U1): 0}

    sample = build_sampler(gt=gt).sample(exclusions=[(T1, U1)])

    assert sample.exclusions.hard_exclusions == frozenset({frozenset({T1, U1})})
    assert sample.exclusions.unknown_pairs == ()


def test_demoted_bridge_hard_bans_the_exact_link_pair():
    sample = build_sampler().sample(exclusions=[(T1, BRIDGE)])

    assert T1 not in {triplet.target for triplet in sample.triplets}
    assert {triplet.target for triplet in sample.triplets} == {T2}
    assert not any(triplet.soft_excluded for triplet in sample.triplets)
    assert sample.exclusions.demoted_bridges == frozenset({BRIDGE})


def test_hard_ban_matches_reversed_pair_orientation():
    gt = {(BRIDGE, H1): 1, (H2, BRIDGE): 1, (T1, BRIDGE): 2, (T2, BRIDGE): 2}

    sample = build_sampler(gt=gt).sample(exclusions=[(H1, BRIDGE)])

    assert H1 not in {triplet.home for triplet in sample.triplets}
    assert {triplet.home for triplet in sample.triplets} == {H2}


def test_soft_bridge_flags_all_its_triplets():
    sample = build_sampler().sample(exclusions=[(T1, BRIDGE), (T2, BRIDGE)])

    assert len(sample.triplets) == 4
    assert all(triplet.soft_excluded for triplet in sample.triplets)
    assert sample.exclusions.avoided_bridges == frozenset({BRIDGE})
    assert sample.exclusions.demoted_bridges == frozenset()


def test_soft_bridge_flags_triplets_when_all_home_pairs_are_excluded():
    sample = build_sampler().sample(exclusions=[(H1, BRIDGE), (H2, BRIDGE)])

    assert all(triplet.soft_excluded for triplet in sample.triplets)
    assert sample.exclusions.avoided_bridges == frozenset({BRIDGE})


def test_unknown_exclusion_pair_is_reported_and_ignored():
    baseline = build_sampler().sample()

    sample = build_sampler().sample(exclusions=[(U1, T2)])

    assert sample.exclusions.unknown_pairs == ((U1, T2),)
    assert sample.triplets == baseline.triplets


def test_unknown_exclusion_pair_with_unknown_record_is_reported():
    stranger = RefId(999, OTHER)

    sample = build_sampler().sample(exclusions=[(stranger, BRIDGE)])

    assert sample.exclusions.unknown_pairs == ((stranger, BRIDGE),)
    assert sample.exclusions.avoided_bridges == frozenset()


def test_exclusion_pair_with_two_bridges_excludes_both():
    bridge_2 = RefId(101, GENERATED)
    gt = {
        **DIRECTED_GT,
        (H1, bridge_2): 1,
        (T2, bridge_2): 2,
        (BRIDGE, bridge_2): 1,
    }
    clusters = {**CLUSTERS, bridge_2: 1}
    refs = REFS + (bridge_2,)

    sample = build_sampler(gt=gt, clusters=clusters, refs=refs).sample(
        exclusions=[(BRIDGE, bridge_2)]
    )

    assert sample.exclusions.demoted_bridges == frozenset({BRIDGE, bridge_2})
    assert sample.exclusions.avoided_bridges == frozenset()


def test_sample_groups_triplets_by_bridge():
    sample = build_sampler().sample()

    assert sample.by_bridge == {BRIDGE: sample.triplets}


def test_source_and_label_two_predicates_produce_the_same_sample():
    directed = build_sampler(
        gt=DIRECTED_GT, is_bridge=by_label_two(DIRECTED_GT)
    ).sample()
    binary = build_sampler(gt=BINARY_GT, is_bridge=by_source).sample()

    assert directed.triplets == binary.triplets


@pytest.mark.parametrize("dataset", DATASETS)
def test_real_dataset_sample_contains_only_declared_structure(data_dir, dataset):
    table, clusters, gt = load_dataset(data_dir / dataset)

    sample = BadTripletSampler(table, gt, clusters, by_source).sample()

    assert sample.triplets
    for triplet in sample.triplets:
        assert clusters[triplet.home] == clusters[triplet.bridge]
        assert clusters[triplet.target] != clusters[triplet.bridge]
        assert declared(gt, (triplet.home, triplet.bridge))
        assert declared(gt, (triplet.target, triplet.bridge))


@pytest.mark.parametrize("dataset", DATASETS)
def test_real_dataset_sample_is_deterministic(data_dir, dataset):
    table, clusters, gt = load_dataset(data_dir / dataset)

    def sample():
        return BadTripletSampler(table, gt, clusters, by_source).sample()

    assert sample().triplets == sample().triplets
