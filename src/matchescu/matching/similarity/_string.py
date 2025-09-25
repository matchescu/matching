from abc import ABCMeta, abstractmethod
from typing import Any, Union, Mapping, Iterable

import numpy as np
from jellyfish import (
    jaccard_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
)
from matchescu.matching.similarity._common import Similarity
from matchescu.matching.similarity._bucketed import BucketedSimilarity


class StringSimilarity(Similarity[float], metaclass=ABCMeta):
    def __init__(
        self,
        ignore_case: bool = False,
        missing_both: float = 0.0,
        missing_one: float = 0.0,
    ):
        super().__init__(missing_both, missing_one)
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


class BucketedLevenshteinDistance(BucketedSimilarity):
    def __init__(
        self,
        buckets: Union[Mapping[float, float], Iterable[float]],
        catch_all: float = 0.0,
        missing_both: float = -1.0,
        missing_either: float = -0.5,
        ignore_case: bool = False,
    ) -> None:
        super().__init__(
            LevenshteinDistance(ignore_case, missing_both, missing_either),
            buckets,
            catch_all,
            missing_both,
            missing_either,
        )


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


class BucketedLevenshteinSimilarity(BucketedSimilarity):
    # the wrapped similarity is in [0, 1]
    __BUCKETS = np.linspace(0.0, 1.0, 11).tolist()
    __CATCH_ALL = __MISSING_BOTH = -1.0
    __MISSING_EITHER = -0.5

    def __init__(
        self,
        ignore_case: bool = False,
    ) -> None:
        super().__init__(
            LevenshteinSimilarity(
                ignore_case, self.__MISSING_BOTH, self.__MISSING_EITHER
            ),
            self.__BUCKETS,
            self.__CATCH_ALL,
            self.__MISSING_BOTH,
            self.__MISSING_EITHER,
        )


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
