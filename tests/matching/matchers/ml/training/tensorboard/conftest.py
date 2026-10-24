from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from tensorboard.compat.proto import (
    event_pb2,
    summary_pb2,
    tensor_pb2,
    tensor_shape_pb2,
    types_pb2,
)
from tensorboard.summary.writer.event_file_writer import EventFileWriter

EventMap = dict[str, Iterable[tuple[int, float]]]
ValueFactory = Callable[[str, float], summary_pb2.Summary.Value]


def _shape(*dimensions: int) -> tensor_shape_pb2.TensorShapeProto:
    return tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in dimensions]
    )


def _simple_value(tag: str, value: float) -> summary_pb2.Summary.Value:
    return summary_pb2.Summary.Value(tag=tag, simple_value=value)


def _scalar_tensor(tag: str, value: float) -> summary_pb2.Summary.Value:
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_FLOAT,
        tensor_shape=_shape(),
        float_val=[value],
    )
    return summary_pb2.Summary.Value(tag=tag, tensor=tensor)


def _write_events(log_dir: Path, events: EventMap, factory: ValueFactory) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(log_dir))
    wall_time = 0.0
    try:
        for tag, tag_events in events.items():
            for step, value in tag_events:
                wall_time += 1.0
                event = event_pb2.Event(
                    wall_time=wall_time,
                    step=step,
                    summary=summary_pb2.Summary(value=[factory(tag, value)]),
                )
                writer.add_event(event)
    finally:
        writer.flush()
        writer.close()


def write_scalar_events(log_dir: Path, events: EventMap) -> None:
    _write_events(log_dir, events, _simple_value)


def write_tensor_events(log_dir: Path, events: EventMap) -> None:
    _write_events(log_dir, events, _scalar_tensor)


def _write_tensor_event(log_dir: Path, tag: str, tensor) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    event = event_pb2.Event(
        wall_time=1.0,
        step=1,
        summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, tensor=tensor)]
        ),
    )
    writer = EventFileWriter(str(log_dir))
    try:
        writer.add_event(event)
    finally:
        writer.flush()
        writer.close()


def write_vector_event(log_dir: Path, tag: str, values: list[float]) -> None:
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_FLOAT,
        tensor_shape=_shape(len(values)),
        float_val=values,
    )
    _write_tensor_event(log_dir, tag, tensor)


def write_string_tensor_event(log_dir: Path, tag: str, value: str) -> None:
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        tensor_shape=_shape(),
        string_val=[value.encode()],
    )
    _write_tensor_event(log_dir, tag, tensor)


@pytest.fixture
def standard_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "standard"
    write_scalar_events(
        log_dir / "run1",
        {
            "loss": [(1, 0.9), (2, 0.7)],
            "accuracy": [(1, 0.5), (2, 0.8)],
        },
    )
    return log_dir


@pytest.fixture
def metric_dir_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "metric-dirs"
    write_scalar_events(log_dir / "model_loss", {"model": [(1, 0.9), (2, 0.7)]})
    write_scalar_events(log_dir / "model_accuracy", {"model": [(1, 0.5), (2, 0.8)]})
    return log_dir


@pytest.fixture
def multi_run_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "multi-run"
    write_scalar_events(log_dir / "model-a_loss", {"model-a": [(1, 0.9)]})
    write_scalar_events(log_dir / "model-b_loss", {"model-b": [(1, 0.4)]})
    return log_dir


@pytest.fixture
def missing_metric_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "missing-metric"
    write_scalar_events(
        log_dir / "run1",
        {
            "loss": [(1, 0.9), (2, 0.7)],
            "accuracy": [(2, 0.8)],
        },
    )
    return log_dir


@pytest.fixture
def tensor_scalar_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "tensor-scalar"
    write_tensor_events(log_dir / "run1", {"loss": [(1, 0.9), (2, 0.7)]})
    return log_dir


@pytest.fixture
def vector_tensor_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "vector-tensor"
    write_vector_event(log_dir / "run1", "vector", [1.0, 2.0])
    return log_dir


@pytest.fixture
def duplicate_step_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "duplicate-step"
    write_scalar_events(log_dir / "run1", {"loss": [(1, 0.5), (1, 0.9)]})
    return log_dir


@pytest.fixture
def string_tensor_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "string-tensor"
    write_string_tensor_event(log_dir / "run1", "label", "not-a-number")
    return log_dir


@pytest.fixture
def nested_metric_dir_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "nested-metric-dirs"
    write_scalar_events(log_dir / "experiment" / "model_loss", {"model": [(1, 0.9)]})
    return log_dir
