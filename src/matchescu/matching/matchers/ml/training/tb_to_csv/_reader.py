from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path

import polars as pl
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.util import tensor_util

_EVENT_FILE_PATTERN = "events.out.tfevents.*"
_RUN_COLUMN = "run"
_STEP_COLUMN = "step"
_NULL_VALUE = ""

_Records = dict[tuple[str, int, str], tuple[float, float]]
_ScalarEvent = tuple[str, int, float, float]


class TensorBoardReader:
    """Read scalar metrics from a TensorBoard log directory.

    Events are grouped by run and step. One column is produced per metric and
    cells without a value are left null. A run is the relative log directory,
    unless a directory name is ``<tag>_<metric>``: then the event tag is the
    run and the directory suffix is the metric. The latter handles writers
    that create one directory per metric.
    """

    def __init__(self, log_dir: str | PathLike) -> None:
        self.__log_dir = Path(log_dir)

    def read(self) -> pl.DataFrame:
        """Return all scalar metrics as a wide polars DataFrame."""
        return _build_frame(self.__load_records())

    def write_csv(self, output_path: str | PathLike) -> None:
        """Write all scalar metrics to a CSV file with empty missing cells."""
        self.read().write_csv(output_path, null_value=_NULL_VALUE)

    def __load_records(self) -> _Records:
        records: _Records = {}
        for event_dir in self.__event_dirs():
            self.__read_event_dir(event_dir, records)
        return records

    def __event_dirs(self) -> list[Path]:
        if not self.__log_dir.exists():
            raise FileNotFoundError(
                f"TensorBoard log directory not found: {self.__log_dir}"
            )
        if not self.__log_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.__log_dir}")
        event_dirs = sorted(
            {path.parent for path in self.__log_dir.rglob(_EVENT_FILE_PATTERN)}
        )
        if not event_dirs:
            raise FileNotFoundError(
                f"No TensorBoard event files found in {self.__log_dir}"
            )
        return event_dirs

    def __read_event_dir(self, event_dir: Path, records: _Records) -> None:
        relative_dir = event_dir.relative_to(self.__log_dir)
        for tag, step, wall_time, value in _scalar_events(_accumulator(event_dir)):
            run, metric = _split_run_and_metric(relative_dir, tag)
            _store_latest(records, run, metric, step, wall_time, value)


def _accumulator(event_dir: Path) -> event_accumulator.EventAccumulator:
    accumulator = event_accumulator.EventAccumulator(
        str(event_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    accumulator.Reload()
    return accumulator


def _scalar_events(
    accumulator: event_accumulator.EventAccumulator,
) -> Iterator[_ScalarEvent]:
    tags = accumulator.Tags()
    for tag in tags.get(event_accumulator.SCALARS, []):
        yield from _scalar_tag_events(tag, accumulator.Scalars(tag))
    for tag in tags.get(event_accumulator.TENSORS, []):
        yield from _tensor_tag_events(tag, accumulator.Tensors(tag))


def _scalar_tag_events(
    tag: str, events: Iterable[event_accumulator.ScalarEvent]
) -> Iterator[_ScalarEvent]:
    for event in events:
        yield tag, event.step, event.wall_time, event.value


def _tensor_tag_events(
    tag: str, events: Iterable[event_accumulator.TensorEvent]
) -> Iterator[_ScalarEvent]:
    for event in events:
        value = _tensor_scalar(event.tensor_proto)
        if value is not None:
            yield tag, event.step, event.wall_time, value


def _tensor_scalar(tensor: TensorProto) -> float | None:
    array = tensor_util.make_ndarray(tensor)
    if array.size != 1:
        return None
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _split_run_and_metric(relative_dir: Path, tag: str) -> tuple[str, str]:
    if not tag:
        return relative_dir.as_posix(), tag
    prefix = f"{tag}_"
    for depth, part in enumerate(relative_dir.parts):
        if not part.startswith(prefix):
            continue
        run = (Path(*relative_dir.parts[:depth]) / tag).as_posix()
        return run, part[len(prefix) :]
    return relative_dir.as_posix(), tag


def _store_latest(
    records: _Records,
    run: str,
    metric: str,
    step: int,
    wall_time: float,
    value: float,
) -> None:
    key = (run, step, metric)
    current = records.get(key)
    if current is None or wall_time >= current[0]:
        records[key] = (wall_time, value)


def _build_frame(records: _Records) -> pl.DataFrame:
    metrics = sorted({metric for _, _, metric in records})
    index = sorted({(run, step) for run, step, _ in records})
    data: dict[str, list] = {
        _RUN_COLUMN: [run for run, _ in index],
        _STEP_COLUMN: [step for _, step in index],
    }
    for metric in metrics:
        data[metric] = [_value_at(records, run, step, metric) for run, step in index]
    schema = {
        _RUN_COLUMN: pl.String,
        _STEP_COLUMN: pl.Int64,
        **{metric: pl.Float64 for metric in metrics},
    }
    return pl.DataFrame(data, schema=schema)


def _value_at(records: _Records, run: str, step: int, metric: str) -> float | None:
    record = records.get((run, step, metric))
    return record[1] if record is not None else None
