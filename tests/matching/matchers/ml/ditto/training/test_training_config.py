import pytest

from matchescu.matching.matchers.ml.ditto.training._training_config import (
    DittoTrainingConfig,
)


@pytest.fixture
def config_file(data_dir) -> str:
    return data_dir / "config.json"


def test_load_from_json(config_file):
    config = DittoTrainingConfig.load_json(config_file)

    assert config.learning_rate == 0.001
    assert config.batch_size == 31
    assert len(config.dataset_configs) == 1
    assert config.dataset_configs["abt-buy"].batch_size == 67
    assert len(config.dataset_configs["abt-buy"].model_configs) == 1
    assert config.dataset_configs["abt-buy"].model_configs["roberta-base"].epochs == 1
    assert len(config.model_configs) == 1
    assert config.model_configs["distilbert-base-uncased"].epochs == 2
