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

    assert len(data.train) > 0


def test_split_with_max_comparison_space(affils_dir, affils_id_col, affils_traits):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping("affiliationstrings_mapping.csv")
        .with_clusters()
    )

    data.split(max_sample_count=10000)

    assert len(data.train) + len(data.test) + len(data.valid) == 10000
