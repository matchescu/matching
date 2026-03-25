import json
from dataclasses import dataclass, field, asdict
from os import PathLike
from pathlib import Path
from typing import Type, Optional, cast

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
        cls, config: dict, parent: T | None, clazz: Type[T]
    ) -> T:
        lr = 3e-5 if parent is None else parent.learning_rate
        fl = 0 if parent is None else parent.frozen_layer_count
        ep = 10 if parent is None else parent.epochs
        bs = 32 if parent is None else parent.batch_size
        return clazz(
            learning_rate=(float(config.get("learningRate", lr))),
            frozen_layer_count=(int(config.get("frozenLayers", fl))),
            epochs=(int(config.get("epochs", ep))),
            batch_size=(int(config.get("batchSize", bs))),
        )

    @classmethod
    def __read_dataset_training_params[T: DatasetTrainingParams](
        cls, config: dict, parent: T | None, clazz: Type[T]
    ) -> T:
        result = cls.__read_model_training_params(config, parent, clazz)
        if "modelConfig" in config:
            result.model_configs = {
                key: cls.__read_model_training_params(
                    config_node, result, ModelTrainingParams
                )
                for key, config_node in config["modelConfig"].items()
            }
        return result

    @classmethod
    def load_json(cls, f: str | PathLike) -> "TrainingConfig":
        with open(f, "r") as fp:
            config = json.load(fp)

        result = cls.__read_dataset_training_params(config, None, TrainingConfig)
        result.dataset_names = list(map(str, config.get("datasets", [])))
        result.model_names = list(map(str, config.get("models", [])))
        if "datasetConfig" in config:
            result.dataset_configs = {
                key: cls.__read_dataset_training_params(
                    config_node, result, DatasetTrainingParams
                )
                for key, config_node in config["datasetConfig"].items()
            }
        return result

    @staticmethod
    def __attr_val[T](params: ModelTrainingParams, attr_name: str) -> T:
        return cast(T, params.__getattribute__(attr_name))

    def _get_attr[T](
        self, attr_name: str, model: str, dataset: str, default_val: T
    ) -> T:
        cfg = self.dataset_configs.get(dataset, self)
        val = self.__attr_val(cfg, attr_name)
        return val if val is not None and val != default_val else default_val

    def get(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> ModelTrainingParams:
        cfg = self.dataset_configs.get(dataset, self)
        if model is None:
            return cfg

        if model in cfg.model_configs:
            return cfg.model_configs[model]

        if model in self.model_configs:
            base_cfg = asdict(self)
            local_cfg = asdict(cfg)
            remote_cfg = asdict(self.model_configs[model])
            out = {}
            for k, v in local_cfg.items():
                if k == "model_configs":
                    continue
                base_v = base_cfg.get(k, v)
                remote_v = remote_cfg.get(k, v)
                if remote_v != v and remote_v != base_v:
                    out[k] = remote_v
                else:
                    out[k] = v
            cfg = ModelTrainingParams(**out)
        return cfg
