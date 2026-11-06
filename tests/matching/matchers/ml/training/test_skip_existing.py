from types import SimpleNamespace
from typing import ClassVar

import pytest
from click.testing import CliRunner

from matchescu.matching.matchers.ml.core import ModelTrainingParams
from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams
from matchescu.matching.matchers.ml.multiclass.training import MultiClassTrainer
from matchescu.matching.matchers.ml.training.__main__ import run_training

DATASET = "dataset"
MODEL = "model"


class _FakeBuilder:
    def load_data(self):
        return self

    def load_splits(self):
        return self

    def create(self):
        return SimpleNamespace(name=DATASET)


class _FakeConfig:
    included_datasets: ClassVar[list[str]] = [DATASET]
    model_names: ClassVar[list[str]] = [MODEL]
    data_builders: ClassVar[dict[str, _FakeBuilder]] = {DATASET: _FakeBuilder()}

    def get(self, model=None, dataset=None):
        return ModelTrainingParams(model_name=MODEL)

    def get_trainer(self, model):
        return MultiClassTrainer

    def get_evaluator(self, model):
        return object


@pytest.fixture
def cli_args(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    return [
        "-M",
        str(tmp_path),
        "-D",
        str(tmp_path),
        "-f",
        str(config_file),
    ]


@pytest.fixture
def trained_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.training.__main__.TrainingConfig.load_json",
        staticmethod(lambda *args, **kwargs: _FakeConfig()),
    )
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.training.__main__._new_fast_tokenizer",
        lambda name: None,
    )
    monkeypatch.setattr(
        "matchescu.matching.matchers.ml.training.__main__.train_on_benchmark_data",
        lambda *args, **kwargs: calls.append(args),
    )
    return calls


def test_skip_existing_skips_training_when_checkpoint_present(
    tmp_path, cli_args, trained_calls
):
    checkpoint = tmp_path / DATASET / MODEL / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = CliRunner().invoke(run_training, cli_args + ["--skip-existing"])

    assert result.exit_code == 0, result.output
    assert trained_calls == []


def test_without_flag_trains_despite_existing_checkpoint(
    tmp_path, cli_args, trained_calls
):
    checkpoint = tmp_path / DATASET / MODEL / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = CliRunner().invoke(run_training, cli_args)

    assert result.exit_code == 0, result.output
    assert len(trained_calls) == 1


def test_skip_existing_trains_when_checkpoint_missing(
    tmp_path, cli_args, trained_calls
):
    result = CliRunner().invoke(run_training, cli_args + ["--skip-existing"])

    assert result.exit_code == 0, result.output
    assert len(trained_calls) == 1


def test_trainer_model_checkpoint_path_points_into_task_model_dir(
    tmp_path, make_trainer
):
    trainer = MultiClassTrainer(MODEL, MultiClassTrainingParams(), model_dir=tmp_path)

    assert trainer.model_checkpoint_path == tmp_path / MODEL / "model.pt"
    assert trainer.task_model_dir == tmp_path / MODEL
    assert trainer.task_model_dir.is_dir()
