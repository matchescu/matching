import json
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Type, Optional


DEFAULT_MODEL_DIR = Path.cwd() / "models"
DEFAULT_DATA_DIR = Path.cwd() / "data"


@dataclass
class ModelTrainingParams:
    learning_rate: float = field(default=3e-5)
    frozen_layer_count: int = field(default=0)
    epochs: int = field(default=10)
    batch_size: int = field(default=32)


@dataclass
class DatasetTrainingParams(ModelTrainingParams):
    model_configs: dict[str, ModelTrainingParams] = field(default_factory=dict)


@dataclass
class TrainingConfig(DatasetTrainingParams):
    model_names: list[str] = field(default_factory=list)
    dataset_names: list[str] = field(default_factory=list)
    dataset_configs: dict[str, DatasetTrainingParams] = field(default_factory=dict)

    @classmethod
    def __read_model_training_params[T: ModelTrainingParams](
        cls, config: dict, clazz: Type[T]
    ) -> T:
        return clazz(
            learning_rate=(float(config.get("learningRate", 3e-5))),
            frozen_layer_count=(int(config.get("frozenLayers", 0))),
            epochs=(int(config.get("epochs", 10))),
            batch_size=(int(config.get("batchSize", 32))),
        )

    @classmethod
    def __read_dataset_training_params[T: DatasetTrainingParams](
        cls, config: dict, clazz: Type[T]
    ) -> T:
        result = cls.__read_model_training_params(config, clazz)
        if "modelConfig" in config:
            result.model_configs = {
                key: cls.__read_model_training_params(config_node, ModelTrainingParams)
                for key, config_node in config["modelConfig"].items()
            }
        return result

    @classmethod
    def load_json(cls, f: str | PathLike) -> "TrainingConfig":
        with open(f, "r") as fp:
            config = json.load(fp)

        result = cls.__read_dataset_training_params(config, TrainingConfig)
        result.dataset_names = list(map(str, config.get("datasets", [])))
        result.model_names = list(map(str, config.get("models", [])))
        if "datasetConfig" in config:
            result.dataset_configs = {
                key: cls.__read_dataset_training_params(
                    config_node, DatasetTrainingParams
                )
                for key, config_node in config["datasetConfig"].items()
            }
        return result

    def get(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> ModelTrainingParams:
        cfg = self.dataset_configs.get(dataset, self)
        if model is not None:
            cfg = cfg.model_configs.get(model, self.model_configs.get(model, self))
        return cfg
