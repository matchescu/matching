"""
Capability-aware training configuration loader.

Trainers/evaluators register themselves by subclassing BaseTrainer/BaseEvaluator
with a ``capability="..."`` keyword.  At load time the registry is populated via
**``pkgutil``** discovery, and each model variant in defined in the config file
is resolved to a (trainer, evaluator, hyperparams_schema) triple.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Type

from pydantic import TypeAdapter
from pydantic.alias_generators import to_camel

from matchescu.matching.evaluation.data.benchmark._base import BenchmarkDataBuilder
from matchescu.matching.matchers.ml.core import ModelTrainingParams
from matchescu.matching.config import (
    AnyDatasetConfig,
    MagellanBenchmarkDataConfig,
    CsvBenchmarkDataConfig,
)
from matchescu.matching.evaluation.data.benchmark._magellan import (
    MagellanBenchmarkDataBuilder,
)
from matchescu.matching.evaluation.data.benchmark._csv import CsvBenchmarkDataBuilder
from ._evaluator import BaseEvaluator
from ._exceptions import ConfigurationError
from ._registry import CapabilityRegistry


# Consts, utilities and classes safe for internal use only
_STRUCTURAL_KEYS = frozenset(
    {
        "kind",
        "datasets",
        "modelVariants",
        "datasetConfig",
        "modelConfig",
        "models",
        "dataSources",
    }
)


def _extract_hyper_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract scalar hyperparameter entries, skipping structural keys and nested dicts."""
    return {
        k: v
        for k, v in raw.items()
        if k not in _STRUCTURAL_KEYS and not isinstance(v, dict)
    }


class _ResolvedVariant:
    __slots__ = (
        "trainer_cls",
        "evaluator_cls",
        "hyperparams_schema",
        "overrides",
    )

    def __init__(
        self,
        trainer_cls: type,
        evaluator_cls: type,
        hyperparams_schema: Type[ModelTrainingParams],
        overrides: dict[str, Any],
    ) -> None:
        self.trainer_cls = trainer_cls
        self.evaluator_cls = evaluator_cls
        self.hyperparams_schema = hyperparams_schema
        self.overrides = overrides


# Public module interface
DEFAULT_MODEL_DIR = Path.cwd() / "models"
DEFAULT_DATA_DIR = Path.cwd() / "data"
MATCHERS_ML_PACKAGE = "matchescu.matching.matchers.ml"


class TrainingConfig:
    """Manage configuration files and discover trainers and evaluators
    dynamically using the ``CapabilityRegistry``.
    """

    def __init__(self) -> None:
        self._ds_adapter = TypeAdapter(AnyDatasetConfig)
        self._global_hp: dict[str, Any] = {}
        self._variants: dict[str, _ResolvedVariant] = {}
        self._model_hp: dict[str, dict[str, Any]] = {}
        self._model_hp_objs: dict[str, Any] = {}
        self._dataset_hp: dict[str, dict[str, Any]] = {}
        self._ds_hp_objs: dict[str, Any] = {}
        self._dataset_model_hp: dict[str, dict[str, dict[str, Any]]] = {}
        self._data_builders: dict[str, BenchmarkDataBuilder] = {}
        self.included_datasets: list[str] = []
        self.included_models: list[str] = []

    @classmethod
    def load_json(
        cls,
        config_file: str | PathLike,
        *,
        data_dir: Path | None = None,
        discovery_packages: list[str] | None = None,
        default_evaluator: Type[BaseEvaluator] | None = None,
    ) -> TrainingConfig:
        if discovery_packages:
            CapabilityRegistry.discover(*discovery_packages)

        with open(config_file, "r") as fp:
            raw: dict[str, Any] = json.load(fp)

        cfg = cls()
        cfg.included_datasets = list(map(str, raw.get("datasets", [])))
        cfg.included_models = list(map(str, raw.get("models", [])))
        cfg._global_hp = _extract_hyper_params(raw)

        requested_kind = raw.get("kind", "")
        trainer_cls, evaluator_cls = CapabilityRegistry.resolve(
            requested_kind,
            default_evaluator=default_evaluator,
        )
        cfg._schema = getattr(trainer_cls, "hyperparams_schema", ModelTrainingParams)
        cfg._variants = {
            model: _ResolvedVariant(trainer_cls, evaluator_cls, cfg._schema, {})
            for model in cfg.included_models
        }

        raw_model_configs: dict = raw.get("modelConfig", {})
        for model_name, spec in raw_model_configs.items():
            if not isinstance(spec, dict):
                continue
            overrides = spec.copy()
            cfg._model_hp[model_name] = overrides
            cfg._model_hp_objs[model_name] = cfg._schema.model_validate(
                {**cfg._global_hp, **overrides}
            )

        for ds_name, ds_raw in raw.get("datasetConfig", {}).items():
            ds_cfg = _extract_hyper_params(ds_raw)
            cfg._dataset_hp[ds_name] = ds_cfg
            cfg._ds_hp_objs[ds_name] = cfg._schema.model_validate(
                {**cfg._global_hp, **ds_cfg}
            )

            ds_models: dict[str, dict[str, Any]] = {}
            # nested modelConfig block
            ds_model_cfg = ds_raw.pop("modelConfig", {})
            for m, m_raw in ds_model_cfg.items():
                ds_models[m] = _extract_hyper_params(m_raw)
            for k, v in ds_raw.items():
                if k in ds_cfg:
                    continue
                if not isinstance(v, dict):
                    continue
                # Implicit: model names as direct keys at dataset level
                ds_models[k] = _extract_hyper_params(v)

            if ds_models:
                cfg._dataset_model_hp[ds_name] = ds_models

        for ds_name, ds_raw in raw.get("datasets", {}).items():
            params = cfg._ds_adapter.validate_python(ds_raw)
            cfg._data_builders[ds_name] = cfg.new_data_builder(params, data_dir)

        return cfg

    @staticmethod
    def new_data_builder(
        params: AnyDatasetConfig, data_dir: Path | None = None
    ) -> BenchmarkDataBuilder:
        """Given a validated dataset params object, return the right factory."""
        match params:
            case MagellanBenchmarkDataConfig():
                return MagellanBenchmarkDataBuilder(params, data_dir)
            case CsvBenchmarkDataConfig():
                return CsvBenchmarkDataBuilder(params, data_dir)
            case _:
                raise ValueError(f"Unsupported dataset params type: {type(params)}")

    def get[TParams](
        self,
        model: str | None = None,
        dataset: str | None = None,
    ) -> TParams:
        """Resolve hyperparameters with a cascading fallback.

        The order in which settings are resolved is:

        1. dataset + model override (highest priority)
        2. top-level model override, merged with dataset defaults
        3. dataset defaults
        4. global defaults (lowest priority)

        The returned instance is validated against the hyperparameter schema
        advertised by the trainer of the kind specified in the config file.
        """
        base = self._schema.model_validate(self._global_hp).model_dump(by_alias=True)

        # No model requested => dataset or global level
        if model is None:
            if dataset and dataset in self._dataset_hp:
                return self._schema.model_validate(
                    {**self._global_hp, **self._dataset_hp[dataset]}
                )
            return self._schema.model_validate(self._global_hp)

        # Dataset overrides on top of global (or just global)
        ds_hp = self._dataset_hp.get(dataset, {}) if dataset else {}
        local_obj = self._schema.model_validate({**self._global_hp, **ds_hp})
        local = local_obj.model_dump(by_alias=True)

        # If requested dataset has a model-specific override
        if dataset:
            ds_model_hp = self._dataset_model_hp.get(dataset, {}).get(model)
            if ds_model_hp is not None:
                return self._schema.model_validate(
                    {**self._global_hp, **ds_hp, **ds_model_hp}
                )

        # If there's a top-level model override, merge top-level model with ds
        if model in self._model_hp:
            remote = self._schema.model_validate(
                {**self._global_hp, **self._model_hp[model]}
            ).model_dump(by_alias=True)

            merged: dict[str, Any] = {}
            for k, v in local.items():
                base_v = base.get(k, v)
                remote_v = remote.get(k, v)
                # Apply model-specific value only if it was *explicitly*
                # changed (differs from both local and global base)
                if remote_v != v and remote_v != base_v:
                    merged[k] = remote_v
                else:
                    merged[k] = v
            return self._schema.model_validate(merged)

        # If there aren't any model overrides, simply return the dataset hyper-params
        return local_obj

    @property
    def data_builders(self) -> dict[str, BenchmarkDataBuilder]:
        return self._data_builders

    @property
    def model_names(self) -> list[str]:
        return self.included_models

    @property
    def dataset_configs(self) -> dict:
        return self._ds_hp_objs

    @property
    def model_configs(self) -> dict:
        return self._model_hp_objs

    def __dir__(self):
        return list(sorted(self._global_hp.keys()))

    def __getattr__(self, item):
        key = to_camel(item)
        if key not in self._global_hp:
            raise AttributeError(
                f"{self.__class__.__name__} does not have a '{item}' attribute"
            )
        return self._global_hp[key]

    def get_variant(self, model: str) -> _ResolvedVariant | None:
        return self._variants.get(model)

    def __get_variant(self, model: str) -> _ResolvedVariant:
        v = self._variants.get(model)
        if v is None:
            raise ConfigurationError(f"No variant configured for model '{model}'")
        return v

    def get_trainer(self, model: str) -> type:
        return self.__get_variant(model).trainer_cls

    def get_evaluator(self, model: str) -> type:
        return self.__get_variant(model).evaluator_cls

    def get_schema(self, model: str) -> Type[ModelTrainingParams]:
        v = self._variants.get(model)
        return v.hyperparams_schema if v else ModelTrainingParams
