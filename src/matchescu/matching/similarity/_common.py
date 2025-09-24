from abc import abstractmethod, ABCMeta
from typing import Any, TypeVar, Protocol, Generic


T = TypeVar("T")


class SimilarityFunction(Protocol[T]):
    def __call__(self, a: Any, b: Any) -> T:
        pass


class Similarity(Generic[T], SimilarityFunction[T], metaclass=ABCMeta):
    def __init__(self, both_missing: T, either_missing: T):
        self.__b = both_missing
        self.__e = either_missing

    @abstractmethod
    def _compute_similarity(self, a: Any, b: Any) -> T:
        pass

    def __call__(self, a: Any, b: Any) -> T:
        if a is None and b is None:
            return self.__b
        elif a is None or b is None:
            return self.__e
        else:
            return self._compute_similarity(a, b)
