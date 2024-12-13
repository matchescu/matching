from abc import ABCMeta, abstractmethod
from typing import Any

from jellyfish import (
    jaccard_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
)
from matchescu.matching.similarity._common import Similarity


class StringSimilarity(Similarity, metaclass=ABCMeta):
    def __init__(self, ignore_case: bool = False):
        self.__ignore_case = ignore_case

    @abstractmethod
    def _compute_string_similarity(self, x: str, y: str) -> float:
        pass

    def _compute_similarity(self, a: Any, b: Any) -> float:
        x = str(a or "")
        y = str(b or "")

        if self.__ignore_case:
            x = x.lower()
            y = y.lower()

        return self._compute_string_similarity(x, y)


class LevenshteinDistance(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return levenshtein_distance(x, y)


class LevenshteinSimilarity(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        m = len(x)
        n = len(y)

        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0

        relative_distance = levenshtein_distance(x, y) / max(m, n)
        return round(1 - relative_distance, ndigits=2)


class Jaro(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return jaro_similarity(x, y)


class JaroWinkler(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return jaro_winkler_similarity(x, y)


class Jaccard(StringSimilarity):
    def __init__(self, ignore_case: bool = False, threshold: int | None = None):
        super().__init__(ignore_case)
        self.__threshold = threshold

    def _compute_string_similarity(self, x: str, y: str) -> float:
        y_len = len(y)
        x_len = len(x)
        threshold = self.__threshold or min(x_len, y_len)
        if threshold == 0:
            return 0 if x_len > 0 or y_len > 0 else 1

        return jaccard_similarity(x, y, threshold)
