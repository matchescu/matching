from pathlib import Path

import click

from ._reader import TensorBoardReader


@click.command()
@click.argument(
    "log_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
)
@click.argument(
    "output_csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def run_export(log_dir: Path, output_csv: Path) -> None:
    """Write all scalar metrics found in LOG_DIR to OUTPUT_CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    TensorBoardReader(log_dir).write_csv(output_csv)


if __name__ == "__main__":
    run_export()
