from typing import Generic

from pydantic import BaseModel

from matchescu.matching.matchers.ml.core._typevars import TParams


class AdditionalModelInfo(Generic[TParams], BaseModel):
    hyperparameters: TParams
    best_config: dict
