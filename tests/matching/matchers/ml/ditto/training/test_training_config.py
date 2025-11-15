import pytest

from matchescu.matching.matchers.ml.ditto.training._training_config import (
    DittoTrainingConfig,
)


@pytest.fixture
def config_file(data_dir) -> str:
    return data_dir / "config.json"


@pytest.fixture
def config(config_file) -> DittoTrainingConfig:
    return DittoTrainingConfig.load_json(config_file)


def test_load_from_json(config):
    assert config.learning_rate == 0.001
    assert config.batch_size == 31
    assert len(config.dataset_configs) == 1
    assert config.dataset_configs["abt-buy"].batch_size == 67
    assert len(config.dataset_configs["abt-buy"].model_configs) == 1
    assert config.dataset_configs["abt-buy"].model_configs["roberta-base"].epochs == 1
    assert len(config.model_configs) == 1
    assert config.model_configs["distilbert-base-uncased"].epochs == 2


@pytest.mark.parametrize(
    "model,expected", [("distilbert-base-uncased", 2), ("roberta-base", 10)]
)
def test_load_model_setting_cascade(config, model, expected):
    assert config.get("epochs", model=model) == expected


@pytest.mark.parametrize("dataset,expected", [("abt-buy", 67), ("anything", 31)])
def test_load_dataset_setting_cascade(config, dataset, expected):
    assert config.get("batch_size", dataset=dataset) == expected


@pytest.mark.parametrize(
    "dataset,model,expected",
    [
        ("abt-buy", "roberta-base", 1),
        ("abt-buy", "distilbert-base-uncased", 2),
        ("anything", "roberta-base", 10),
    ],
)
def test_load_model_per_dataset_setting_cascade(config, model, dataset, expected):
    assert config.get("epochs", model, dataset) == expected
