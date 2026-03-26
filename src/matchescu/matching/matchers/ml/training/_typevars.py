from typing import TypeVar, Sized, Union

from torch.nn import Module
from torch.utils.data import Dataset

from ._params import ModelTrainingParams


TModel = TypeVar("TModel", bound=Module)
TParams = TypeVar("TParams", bound=ModelTrainingParams)
TDataset = TypeVar("TDataset", bound=Union[Sized, Dataset])
