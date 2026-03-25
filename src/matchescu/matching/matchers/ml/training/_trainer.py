from abc import ABC, abstractmethod
from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import ClassVar, Type, Any, Generic, cast

from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from ._evaluator import BaseEvaluator
from ._params import ModelTrainingParams
from ._registry import CapabilityRegistry
from ._typevars import TModel, TParams


class BaseTrainer(ABC, Generic[TModel, TParams]):
    """
    Subclass with ``capability="xxx"`` to auto-register::

        class DittoTrainer(BaseTrainer, capability="ditto"):
            hyperparams_schema = DittoHyperParams

            def train(self, model_name, hyperparams): ...
    """

    capability: ClassVar[str] = ""
    hyperparams_schema: ClassVar[Type[ModelTrainingParams]] = ModelTrainingParams

    def __init_subclass__(cls, capability: str = "", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if capability:
            cls.capability = capability
            CapabilityRegistry.register_trainer(capability, cls)

    def __init__(
        self,
        task: str,
        model_dir: str | PathLike,
        hyper_params: TParams,
        loss_fn: _Loss,
        **kwargs: Any,
    ) -> None:
        self._task = task
        self._params = hyper_params
        self._model_dir = Path(model_dir)
        self._loss = loss_fn
        self._log = cast(
            Logger, kwargs.get("logger", getLogger(self.__class__.__name__))
        ).getChild(self._task)

    @abstractmethod
    def run_training(
        self,
        model: TModel,
        training_data: DataLoader,
        evaluator: BaseEvaluator,
        save_model: bool,
        **kwargs: Any,
    ) -> Any:
        pass
