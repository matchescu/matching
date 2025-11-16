import pytest

from matchescu.matching.matchers.ml.ditto.training._config import TrainingConfig


@pytest.fixture
def config_file(data_dir) -> str:
    return data_dir / "config.json"


@pytest.fixture
def config(config_file) -> TrainingConfig:
    return TrainingConfig.load_json(config_file)


def test_load_from_json(config):
    assert config.model_names == ["roberta-base", "distilbert-base-uncased"]
    assert config.dataset_names == ["abt-buy"]
    assert config.learning_rate == 0.001
    assert config.batch_size == 31
    assert len(config.dataset_configs) == 2
    abt_buy_cfg = config.dataset_configs["abt-buy"]
    assert abt_buy_cfg.batch_size == 67
    assert len(abt_buy_cfg.model_configs) == 1
    assert abt_buy_cfg.model_configs["roberta-base"].epochs == 1
    fz_config = config.dataset_configs["fodors-zagat"]
    assert fz_config.frozen_layer_count == 0
    assert fz_config.learning_rate == 0.001
    assert fz_config.batch_size == 32
    assert fz_config.epochs == 3
    assert len(config.model_configs) == 1
    assert config.model_configs["distilbert-base-uncased"].epochs == 2


@pytest.mark.parametrize(
    "model,expected", [("distilbert-base-uncased", 2), ("roberta-base", 15)]
)
def test_load_model_setting_cascade(config, model, expected):
    assert config.get(model=model).epochs == expected


@pytest.mark.parametrize("dataset,expected", [("abt-buy", 67), ("anything", 31)])
def test_load_dataset_setting_cascade(config, dataset, expected):
    assert config.get(dataset=dataset).batch_size == expected


@pytest.mark.parametrize(
    "dataset,model,expected",
    [
        ("abt-buy", "roberta-base", 1),
        ("abt-buy", "distilbert-base-uncased", 2),
        ("anything", "roberta-base", 15),
    ],
)
def test_load_model_per_dataset_setting_cascade(config, model, dataset, expected):
    assert config.get(model, dataset).epochs == expected


def test_fodors_zagat(config):
    train_params = config.get(
        dataset="fodors-zagat", model="huawei-noah/TinyBERT_General_6L_768D"
    )

    assert train_params.batch_size == 32
    assert train_params.epochs == 3
    assert train_params.learning_rate == 0.001
    assert train_params.frozen_layer_count == 0
