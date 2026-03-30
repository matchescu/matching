from pydantic import Field

from matchescu.matching.matchers.ml.training._config_model import ConfigModel


class ModelTrainingParams(ConfigModel):
    """
    Base hyperparameter configuration.

    Trainer implementations specify a specific subclass of this model by setting
    the ``hyperparams_schema`` to a subclass of this model. This is the
    mechanism to introduce custom hyperparameter specifications (e.g.
    ``frozen_layer_count`` for fine-tuning transformers).

    :param learning_rate: the learning rate at the start of training
    :type learning_rate: float
    :param epochs: max number of epochs to run the training
    :type epochs: int
    :param batch_size: split the training data into batches of this size
    :type batch_size: int
    """

    learning_rate: float = Field(default=3e-5)
    epochs: int = Field(default=10)
    batch_size: int = Field(default=32)
    model_name: str | None = None
