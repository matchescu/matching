import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from os import PathLike
from pathlib import Path
from typing import ClassVar, Any, Generic

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ._registry import CapabilityRegistry
from ._typevars import TModel, TDataset


class BaseEvaluator(AbstractContextManager, Generic[TModel, TDataset]):
    capability: ClassVar[str] = ""

    __IS_EVAL_KWARG = "is_evaluating"

    def __init__(
        self,
        task_name: str,
        dev_data: DataLoader[TDataset],
        test_data: DataLoader[TDataset],
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
    def _run_model(
        self, model: TModel, data: DataLoader[TDataset], best_config: dict | None = None
    ) -> tuple[bool, dict]:
        """Run the model in evaluation mode on the specified data.

        This method serves to purposes: hyperparameter tuning and model
        evaluation. When invoked for hyperparameter tuning by the ``_tune``
        method, this method must find the optimal configuration of the model
        given the supplied input data. The boolean return value will be
        interpreted as having found the best configuration. The returned mapping
        will contain the best configuration in this case. Here, the 'best'
        configuration is determined factoring in all previous runs as well. If
        a new best configuration was not found then ``False, {}`` must be
        returned.

        When invoked for evaluating the model from the ``_evaluate`` method,
        this method will be called with the best known configuration obtained
        during the ``_tune`` phase. This configuration will be supplied through
        the ``kwargs``. The expected return value in this case is the input
        configuration (passed through ``kwargs``), updated with new information
        from the evaluation phase. The boolean flag should simply indicate
        whether the run was successful or not in this case.

        :param model: the model to run in evaluation mode
        :param data: the data to use as input for the model

        :return: a boolean and a dict containing the model's best configuration.
        """
        pass

    @classmethod
    def _repr_config(cls, value: dict) -> str:
        return ", ".join(f"{k}={v:.4f}" for k, v in value.items())

    def _is_evaluating(self, config: dict | None) -> bool:
        if not config:
            return False
        return bool(config.pop(self.__IS_EVAL_KWARG, False))

    def __call__(
        self, model: TModel, training_metrics: dict, epoch: int
    ) -> tuple[bool, dict]:
        """Find the metadata of the best model version on the dev data.

        Returns information about the model version that performed best on the
        validation/dev split. Also returns that model version's results on the
        test split. The boolean value indicates whether the new call identified
        a better model version compared to the previous call.
        """
        prev_training = model.training
        try:
            model.eval()

            self._log.info("tuning on dev")
            found_new_best, best_config = self._run_model(model, self._xv_data)
            if not found_new_best:
                self._log.info("hyperparameter tuning: no improvements")
                return found_new_best, best_config

            self._log.info("evaluating on test")
            best_config[self.__IS_EVAL_KWARG] = True
            ok, best_config = self._run_model(model, self._test_data, best_config)
            if not ok:
                self._log.warning("failed to evaluate model on test")
                return ok, best_config

            training_metrics.update(best_config)
            self._summary_writer.add_scalars(self._task, training_metrics, epoch)
            self._log.info("tuning succeeded: %s", self._repr_config(best_config))
            return found_new_best, best_config
        finally:
            model.train(prev_training)

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
