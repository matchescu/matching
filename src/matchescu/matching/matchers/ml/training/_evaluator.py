import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from os import PathLike
from pathlib import Path
from typing import ClassVar, Type, Any, Generic

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ._params import ModelTrainingParams
from ._registry import CapabilityRegistry
from ._typevars import TModel, TParams


class BaseEvaluator(AbstractContextManager, Generic[TModel, TParams]):
    capability: ClassVar[str] = ""
    hyperparams_schema: ClassVar[Type[ModelTrainingParams]] = ModelTrainingParams

    def __init__(
        self,
        task_name: str,
        dev_data: DataLoader,
        test_data: DataLoader,
        tb_log_dir: str | PathLike | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._task = task_name
        self._xv_data = dev_data
        self._test_data = test_data
        self._tb_log_dir = Path(tb_log_dir or __file__).absolute()
        self._summary_writer = SummaryWriter(log_dir=str(self._tb_log_dir))
        self._log = (logger or logging.getLogger(self.__class__.__name__)).getChild(
            self._task
        )

    def __init_subclass__(cls, capability: str = "", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if capability:
            cls.capability = capability
            CapabilityRegistry.register_evaluator(capability, cls)

    @property
    def summary_writer(self):
        return self._summary_writer

    @abstractmethod
    def __call__(
        self, model_name: str, training_metrics: dict, hyperparams: TParams
    ) -> tuple[bool, dict]:
        """Find the metadata of the best model version on the dev data.

        Returns information about the model version that performed best on the
        validation/dev split. Also returns that model version's results on the
        test split. The boolean value indicates whether the new call identified
        a better model version compared to the previous call.
        """
        pass

    def __enter__(self) -> "BaseEvaluator":
        if self._summary_writer is not None:
            return self
        self._summary_writer = SummaryWriter(log_dir=str(self._tb_log_dir))
        return self

    def __exit__(self, exc_type, exc_value, traceback, /):
        if self._summary_writer is None:
            return

        self._summary_writer.flush()
        self._summary_writer.close()
        self._summary_writer = None
