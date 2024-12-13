from decimal import Decimal
from typing import Any

from matchescu.matching.similarity import Similarity


class Norm(Similarity):
    __SUPPORTED_TYPES = (int, float, Decimal)

    def _compute_similarity(self, a: Any, b: Any) -> float:
        if not isinstance(a, self.__SUPPORTED_TYPES) or not isinstance(
            b, self.__SUPPORTED_TYPES
        ):
            return 0
        return abs(float(a - b))


class BoundedNumericDifferenceSimilarity(Similarity):
    __SUPPORTED_TYPES = (int, float, Decimal)

    def __init__(self, max_diff: float = 1.0) -> None:
        self.__max_diff = max_diff

    def _compute_similarity(self, a: Any, b: Any) -> float:
        if not isinstance(a, self.__SUPPORTED_TYPES) or not isinstance(
            b, self.__SUPPORTED_TYPES
        ):
            return 0
        diff = abs(a - b)
        if diff > self.__max_diff:
            diff = self.__max_diff
        return 1 - (diff / self.__max_diff)
