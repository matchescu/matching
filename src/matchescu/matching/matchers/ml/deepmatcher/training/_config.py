import json
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Optional, cast

DEFAULT_MODEL_DIR = Path.cwd() / "models"
DEFAULT_DATA_DIR = Path.cwd() / "data"


@dataclass
class TrainingParams:
    learning_rate: float = field(default=3e-5)
    epochs: int = field(default=10)
    batch_size: int = field(default=32)


@dataclass
class TrainingConfig(TrainingParams):
    dataset_names: list[str] = field(default_factory=list)
    dataset_configs: dict[str, TrainingParams] = field(default_factory=dict)

    @classmethod
    def load_json(cls, f: str | PathLike) -> "TrainingConfig":
        with open(f, "r") as fp:
            config = json.load(fp)
        result = TrainingConfig()
        result.dataset_names = list(map(str, config.get("datasets", [])))
        result.learning_rate = float(config.get("learningRate", 3e-5))
        result.batch_size = int(config.get("batchSize", 32))
        result.epochs = int(config.get("epochs", 15))

        for ds_name, params in config.get("options", {}).items():
            result.dataset_configs[ds_name] = TrainingParams(
                learning_rate=float(params.get("learningRate", result.learning_rate)),
                batch_size=int(params.get("batchSize", result.batch_size)),
                epochs=int(params.get("epochs", result.epochs)),
            )
        return result

    @staticmethod
    def __attr_val[T](params: TrainingParams, attr_name: str) -> T:
        return cast(T, params.__getattribute__(attr_name))

    def _get_attr[T](self, attr_name: str, dataset: str, default_val: T) -> T:
        cfg = self.dataset_configs.get(dataset, self)
        val = self.__attr_val(cfg, attr_name)
        return val if val is not None and val != default_val else default_val

    def get(self, dataset: Optional[str] = None) -> TrainingParams:
        return self.dataset_configs.get(dataset, self)
