from typing import TypeVar

from torch.nn import Module

from ._params import ModelTrainingParams

TModel = TypeVar("TModel", bound=Module)
TParams = TypeVar("TParams", bound=ModelTrainingParams)
