import logging
from unittest.mock import MagicMock

import pytest

from matchescu.matching.config import CsvBenchmarkDataConfig
from matchescu.matching.config._dataset_config import SourceFileConfig, SplitConfig
from matchescu.matching.evaluation.data.benchmark._csv import (
    CsvBenchmarkData,
    CsvBenchmarkDataBuilder,
)
from matchescu.matching.evaluation.data.splits import (
    BadTripletSplitGenerator,
    SplitGenerator,
)


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


def test_split_with_bridge_classes_and_max_comparison_space(
    affils_dir, affils_id_col, affils_traits
):
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids_with_source.csv"])
        .load_data([affils_traits], [affils_id_col], ["source"])
        .with_ideal_mapping(
            "directed_with_label.csv",
            id_cols=(0, 2),
            source_cols=(1, 3),
            label_col=4,
            has_header=True,
        )
        .with_clusters("affiliationstrings_clusters.csv")
    )

    data.split(max_sample_count=1000, neg_pos_ratio=4, bridge_class_ratio=2)

    assert len(data.train_split) + len(data.test_split) + len(data.valid_split) == 1000


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
    assert len(data.ref_id_cluster_map) == 0


def test_true_clusters(affils_dir, affils_id_col, affils_traits):
    mapping_file = "affiliationstrings_mapping.csv"
    data = (
        CsvBenchmarkData(affils_dir, ["affiliationstrings_ids.csv"])
        .load_data([affils_traits], [affils_id_col])
        .with_ideal_mapping(mapping_file)
        .with_clusters()
    )

    assert len(data.ref_id_cluster_map) == len(data.id_table)
    cluster_count = len(set(data.ref_id_cluster_map.values()))
    assert cluster_count == 330
    assert cluster_count <= len(data.id_table)


def test_missing_split_files_warn_and_regenerate_to_configured_names(tmp_path, caplog):
    builder = object.__new__(CsvBenchmarkDataBuilder)
    builder._data_dir = tmp_path
    builder._log = logging.getLogger("builder-test")
    builder._instance = MagicMock()
    builder._params = CsvBenchmarkDataConfig(
        directory=tmp_path,
        sources=[SourceFileConfig(file_name="ids.csv")],
        pairwise_mapping="mapping.csv",
        splits=SplitConfig(
            files={
                "train": "train-trial.csv",
                "valid": "valid-trial.csv",
                "test": "test-trial.csv",
            }
        ),
    )

    with caplog.at_level(logging.WARNING, logger="builder-test"):
        builder.load_splits()

    assert "missing" in caplog.text
    builder._instance.split.assert_called_once()
    assert builder._instance.split.call_args.kwargs["file_names"] == {
        "train": "train-trial.csv",
        "valid": "valid-trial.csv",
        "test": "test-trial.csv",
    }
    assert builder._instance.split.call_args.kwargs["split_mode"] == "pair"


@pytest.mark.parametrize(
    "mode,expected",
    [("pair", SplitGenerator), ("bad_triplet", BadTripletSplitGenerator)],
)
def test_split_mode_selects_generator(mode, expected):
    assert CsvBenchmarkData._generator_class(mode) is expected


def test_split_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported split mode"):
        CsvBenchmarkData._generator_class("unknown")
