from typing import Generic

from pydantic import BaseModel

from matchescu.matching.matchers.ml.core._typevars import TParams


class AdditionalModelInfo(BaseModel, Generic[TParams]):
    hyperparameters: TParams
    best_config: dict
