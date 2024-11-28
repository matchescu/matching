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
    def _compute_string_similarity(self, x: str, m: int, y: str, n: int) -> float:
        pass

    def _compute_similarity(self, a: Any, b: Any) -> float:
        x = str(a)
        y = str(b)
        m = len(x)
        n = len(y)

        if m == 0 and n == 0:
            return 1
        if m == 0 or n == 0:
            return 0
        if self.__ignore_case:
            x = x.lower()
            y = y.lower()
        return self._compute_string_similarity(x, m, y, n)


class Levenshtein(StringSimilarity):
    def _compute_string_similarity(self, x: str, m: int, y: str, n: int) -> float:
        d = levenshtein_distance(x, y)
        res = 1 - d / max(m, n)
        return round(res, 2)


class Jaro(StringSimilarity):
    def _compute_string_similarity(self, x: str, m: int, y: str, n: int) -> float:
        return jaro_similarity(x, y)


class JaroWinkler(StringSimilarity):
    def _compute_string_similarity(self, x: str, m: int, y: str, n: int) -> float:
        return jaro_winkler_similarity(x, y)


class Jaccard(StringSimilarity):
    def __init__(self, ignore_case: bool = False, threshold: int | None = None):
        super().__init__(ignore_case)
        self.__threshold = threshold

    def _compute_string_similarity(self, x: str, m: int, y: str, n: int) -> float:
        threshold = self.__threshold or min(m, n)
        return jaccard_similarity(x, y, threshold)
