from types import SimpleNamespace
from typing import ClassVar

from click.testing import CliRunner

from matchescu.matching.matchers.ml.core import ModelTrainingParams
from matchescu.matching.matchers.ml.multiclass.training import MultiClassTrainer
from matchescu.matching.matchers.ml.training.__main__ import run_training


class _FakeBuilder:
    def load_data(self):
        return self

    def load_splits(self):
        return self

    def create(self):
        return SimpleNamespace(name="dataset")


class _FakeConfig:
    included_datasets: ClassVar[list[str]] = ["dataset"]
    model_names: ClassVar[list[str]] = ["model"]
    data_builders: ClassVar[dict[str, _FakeBuilder]] = {"dataset": _FakeBuilder()}

    def get(self, model=None, dataset=None):
        return ModelTrainingParams(model_name="model")

    def get_trainer(self, model):
        return MultiClassTrainer

    def get_evaluator(self, model):
        return object


def test_cli_seed_overrides_config(monkeypatch, tmp_path):
    captured = {}
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
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
        lambda *args, **kwargs: captured.update(seed=args[6].seed),
    )

    result = CliRunner().invoke(
        run_training,
        [
            "-M",
            str(tmp_path),
            "-D",
            str(tmp_path),
            "-f",
            str(config_file),
            "--seed",
            "43",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["seed"] == 43


def test_training_params_accept_a_seed():
    assert ModelTrainingParams(seed=7).seed == 7
