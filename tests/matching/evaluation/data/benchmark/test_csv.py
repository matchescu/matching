import pytest

from matchescu.matching.evaluation.data.benchmark._csv import CsvBenchmarkData


def test_single_source(affils_dir, affils_id_col, affils_traits):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters("affiliationstrings_clusters.csv")
    )

    assert data.comparison_space_size == 2_552_670
    assert data.ideal_mapping_size == 32816
    assert data.cluster_count == 2589


def test_single_source_without_external_cluster_spec(
    affils_dir, affils_id_col, affils_traits
):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters()
    )

    assert data.comparison_space_size == 2_552_670
    assert data.ideal_mapping_size == 32816
    assert data.cluster_count == 2260


def test_single_source_directed_matcher_gt_must_specify_cluster_labels(
    affils_dir, affils_id_col, affils_traits
):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping(
            "directed_with_label.csv", has_header=True, label_col="label"
        )
    )

    with pytest.raises(ValueError) as err_proxy:
        data.with_clusters()

    assert (
        str(err_proxy.value)
        == "missing cluster labels for multi-class pairwise mappings"
    )


def test_split(affils_dir, affils_id_col, affils_traits):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters()
    )

    data.split()

    assert len(data.train_split) > 0


def test_split_with_max_comparison_space(affils_dir, affils_id_col, affils_traits):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters()
    )

    data.split(max_sample_count=10000)

    assert len(data.train_split) + len(data.test_split) + len(data.valid_split) == 10000


def test_true_matches(affils_dir, affils_id_col, affils_traits):
    mapping_file = "affiliationstrings_mapping.csv"
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping(mapping_file)
    )
    with open(affils_dir / mapping_file, "r") as f:
        line_count = len(f.readlines())

    assert len(data.true_matches) == line_count
    assert len(data.true_clusters) == 0


def test_true_clusters(affils_dir, affils_id_col, affils_traits):
    mapping_file = "affiliationstrings_mapping.csv"
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping(mapping_file)
        .with_clusters()
    )

    assert len(data.true_clusters) == len(data.id_table)
    cluster_count = len(set(data.true_clusters.values()))
    assert cluster_count == 330
    assert cluster_count <= len(data.id_table)
