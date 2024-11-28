from abc import abstractmethod, ABCMeta
from typing import Any


class Similarity(metaclass=ABCMeta):
    @abstractmethod
    def _compute_similarity(self, a: Any, b: Any) -> float:
        pass

    def __call__(self, a: Any, b: Any) -> float:
        if a is None and b is None:
            return 1
        elif a is None or b is None:
            return 0
        else:
            return self._compute_similarity(a, b)
