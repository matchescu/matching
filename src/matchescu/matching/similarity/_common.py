from abc import abstractmethod, ABCMeta
from typing import Any, TypeVar, Protocol, Generic


T = TypeVar("T")


class SimilarityFunction(Protocol[T]):
    def __call__(self, a: Any, b: Any) -> T:
        pass


class Similarity(Generic[T], SimilarityFunction[T], metaclass=ABCMeta):
    def __init__(self, both_missing: T, either_missing: T):
        self._miss_both = both_missing
        self._miss_either = either_missing

    @abstractmethod
    def _compute_similarity(self, a: Any, b: Any) -> T:
        pass

    def __call__(self, a: Any, b: Any) -> T:
        if a is None and b is None:
            return self._miss_both
        elif a is None or b is None:
            return self._miss_either
        else:
            return self._compute_similarity(a, b)
