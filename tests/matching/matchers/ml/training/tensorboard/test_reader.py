import polars as pl
import pytest

from matchescu.matching.matchers.ml.training.tb_to_csv import TensorBoardReader


def test_reads_standard_tags_as_metric_columns(standard_log_dir):
    frame = TensorBoardReader(standard_log_dir).read()

    assert frame.columns == ["run", "step", "accuracy", "loss"]
    assert frame["run"].to_list() == ["run1", "run1"]
    assert frame["step"].to_list() == [1, 2]
    assert frame["accuracy"].to_list() == pytest.approx([0.5, 0.8])
    assert frame["loss"].to_list() == pytest.approx([0.9, 0.7])


def test_reads_metric_named_directories_as_metric_columns(metric_dir_log_dir):
    frame = TensorBoardReader(metric_dir_log_dir).read()

    assert frame.columns == ["run", "step", "accuracy", "loss"]
    assert frame["run"].to_list() == ["model", "model"]
    assert frame["accuracy"].to_list() == pytest.approx([0.5, 0.8])
    assert frame["loss"].to_list() == pytest.approx([0.9, 0.7])


def test_keeps_runs_separate(multi_run_log_dir):
    frame = TensorBoardReader(multi_run_log_dir).read()

    assert frame.columns == ["run", "step", "loss"]
    rows = {(run, step): loss for run, step, loss in frame.iter_rows()}
    assert rows[("model-a", 1)] == pytest.approx(0.9)
    assert rows[("model-b", 1)] == pytest.approx(0.4)


def test_leaves_missing_metric_cells_null(missing_metric_log_dir):
    frame = TensorBoardReader(missing_metric_log_dir).read()

    accuracy = frame["accuracy"].to_list()
    assert accuracy[0] is None
    assert accuracy[1] == pytest.approx(0.8)


def test_writes_empty_cells_for_missing_metrics(missing_metric_log_dir, tmp_path):
    output_path = tmp_path / "metrics.csv"

    TensorBoardReader(missing_metric_log_dir).write_csv(output_path)

    cells = output_path.read_text().splitlines()[1].split(",")
    assert cells[:3] == ["run1", "1", ""]
    assert float(cells[3]) == pytest.approx(0.9)


def test_reads_tensor_scalar_events(tensor_scalar_log_dir):
    frame = TensorBoardReader(tensor_scalar_log_dir).read()

    assert frame["loss"].to_list() == pytest.approx([0.9, 0.7])


def test_ignores_non_scalar_tensor_events(vector_tensor_log_dir):
    frame = TensorBoardReader(vector_tensor_log_dir).read()

    assert frame.columns == ["run", "step"]
    assert frame.is_empty()


def test_uses_latest_value_for_duplicate_step(duplicate_step_log_dir):
    frame = TensorBoardReader(duplicate_step_log_dir).read()

    assert frame["loss"].to_list() == pytest.approx([0.9])


def test_reads_nested_metric_named_directories(nested_metric_dir_log_dir):
    frame = TensorBoardReader(nested_metric_dir_log_dir).read()

    assert frame.columns == ["run", "step", "loss"]
    assert frame["run"].to_list() == ["experiment/model"]


def test_ignores_non_numeric_tensor_events(string_tensor_log_dir):
    frame = TensorBoardReader(string_tensor_log_dir).read()

    assert frame.columns == ["run", "step"]
    assert frame.is_empty()


def test_raises_when_log_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        TensorBoardReader(tmp_path / "missing").read()


def test_raises_when_path_is_a_file(tmp_path):
    file_path = tmp_path / "events.out.tfevents.1"
    file_path.write_text("not an event")

    with pytest.raises(NotADirectoryError):
        TensorBoardReader(file_path).read()


def test_raises_when_no_event_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No TensorBoard event files"):
        TensorBoardReader(empty_dir).read()


def test_dataframe_uses_polars(standard_log_dir):
    frame = TensorBoardReader(standard_log_dir).read()

    assert isinstance(frame, pl.DataFrame)
