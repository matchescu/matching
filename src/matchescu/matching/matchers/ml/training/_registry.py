import importlib
import pkgutil
import warnings
from typing import ClassVar

from ._exceptions import ConfigurationError


class CapabilityRegistry:
    """Class-level registry populated by ``__init_subclass__`` hooks on
    ``BaseTrainer`` and ``BaseEvaluator``.
    """

    _trainers: ClassVar[dict[str, list[type]]] = {}
    _evaluators: ClassVar[dict[str, list[type]]] = {}
    _discovered: ClassVar[set[str]] = set()

    @staticmethod
    def __sanitize_string(value: str) -> str:
        if value is None:
            return ""
        return value.strip().lower()

    @classmethod
    def register_trainer(cls, capability: str, klass: type) -> None:
        cls._trainers.setdefault(cls.__sanitize_string(capability), []).append(klass)

    @classmethod
    def register_evaluator(cls, capability: str, klass: type) -> None:
        cls._evaluators.setdefault(cls.__sanitize_string(capability), []).append(klass)

    @classmethod
    def discover(cls, *packages: str) -> None:
        """Import every submodule under *packages* to trigger registrations."""
        for pkg in packages:
            if pkg in cls._discovered:
                continue
            cls._discovered.add(pkg)
            try:
                root = importlib.import_module(pkg)
            except ImportError as exc:
                warnings.warn(f"Cannot import {pkg}: {exc}")
                continue
            if not hasattr(root, "__path__"):
                continue
            for _, modname, _ in pkgutil.walk_packages(
                root.__path__, prefix=root.__name__ + "."
            ):
                try:
                    importlib.import_module(modname)
                except ImportError as exc:
                    warnings.warn(f"Cannot import {modname}: {exc}")

    @classmethod
    def resolve(
        cls,
        capability: str,
        *,
        default_evaluator: type | None = None,
    ) -> tuple[type, type]:
        key = cls.__sanitize_string(capability)
        trainers = cls._trainers.get(key, [])
        evaluators = cls._evaluators.get(key, [])

        if not trainers:
            raise ConfigurationError(
                f"No trainer registered for capability '{capability}'"
            )
        if len(trainers) > 1:
            trainers = "', '".join(t.__qualname__ for t in trainers)
            raise ConfigurationError(
                f"Multiple trainers for '{capability}': '{trainers}'"
            )

        if not evaluators:
            if default_evaluator is None:
                raise ConfigurationError(
                    f"No evaluator for '{capability}' and no default provided"
                )
            evaluator_cls = default_evaluator
        elif len(evaluators) > 1:
            evaluators = "', '".join(e.__qualname__ for e in evaluators)
            raise ConfigurationError(
                f"Multiple evaluators for '{capability}': '{evaluators}'"
            )
        else:
            evaluator_cls = evaluators[0]

        return trainers[0], evaluator_cls

    @classmethod
    def clear(cls) -> None:
        cls._trainers.clear()
        cls._evaluators.clear()
        cls._discovered.clear()
