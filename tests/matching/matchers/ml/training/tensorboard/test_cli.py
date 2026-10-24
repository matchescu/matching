import polars as pl
from click.testing import CliRunner

from matchescu.matching.matchers.ml.training.tb_to_csv.__main__ import run_export


def test_cli_writes_csv(standard_log_dir, tmp_path):
    output_path = tmp_path / "output" / "metrics.csv"

    result = CliRunner().invoke(run_export, [str(standard_log_dir), str(output_path)])

    assert result.exit_code == 0
    assert pl.read_csv(output_path).columns == ["run", "step", "accuracy", "loss"]
