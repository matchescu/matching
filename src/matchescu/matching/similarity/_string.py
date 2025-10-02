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


class BucketedStringSimilarity(BucketedSimilarity):
    # the wrapped similarity is in [0, 1]
    _BUCKETS = [round(float(x), 1) for x in np.linspace(0.0, 1.0, 11)]
    _CATCH_ALL = 0.0
    _MISSING_BOTH = -1.0
    _MISSING_EITHER = -0.5

    def __init__(
        self,
        sim: StringSimilarity,
        buckets: Union[Mapping[float, float], Iterable[float]] | None = None,
        catch_all: float | None = None,
        missing_both: float | None = None,
        missing_either: float | None = None,
    ) -> None:
        super().__init__(
            sim,
            buckets or self._BUCKETS,
            catch_all if catch_all is not None else self._CATCH_ALL,
            missing_both if missing_both is not None else self._MISSING_BOTH,
            missing_either if missing_either is not None else self._MISSING_EITHER,
        )


class LevenshteinDistance(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return levenshtein_distance(x, y)


class BucketedLevenshteinDistance(BucketedStringSimilarity):
    def __init__(
        self,
        buckets: Union[Mapping[float, float], Iterable[float]],
        ignore_case: bool = False,
    ) -> None:
        super().__init__(
            LevenshteinDistance(ignore_case, self._MISSING_BOTH, self._MISSING_EITHER),
            buckets,
            -1.0,
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


class BucketedLevenshteinSimilarity(BucketedStringSimilarity):
    def __init__(self, ignore_case: bool = False) -> None:
        super().__init__(
            LevenshteinSimilarity(ignore_case, self._MISSING_BOTH, self._MISSING_EITHER)
        )


class Jaro(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return jaro_similarity(x, y)


class BucketedJaro(BucketedStringSimilarity):
    def __init__(self, ignore_case: bool = False) -> None:
        super().__init__(Jaro(ignore_case, self._MISSING_BOTH, self._MISSING_EITHER))


class JaroWinkler(StringSimilarity):
    def _compute_string_similarity(self, x: str, y: str) -> float:
        return jaro_winkler_similarity(x, y)


class BucketedJaroWinkler(BucketedStringSimilarity):
    def __init__(
        self,
        buckets: Mapping[float, float] | Iterable[float] | None,
        ignore_case: bool = False,
    ) -> None:
        super().__init__(
            JaroWinkler(ignore_case, self._MISSING_BOTH, self._MISSING_EITHER),
            buckets or self._BUCKETS,
        )


class Jaccard(StringSimilarity):
    def __init__(
        self,
        ignore_case: bool = False,
        ngram_size: int | None = None,
        missing_both: float = 0.0,
        missing_either: float = 0.0,
    ):
        super().__init__(ignore_case, missing_both, missing_either)
        self.__ngram_size = ngram_size

    def _compute_string_similarity(self, x: str, y: str) -> float:
        return jaccard_similarity(x, y, self.__ngram_size)


class BucketedJaccard(BucketedStringSimilarity):
    def __init__(self, ignore_case: bool = False, threshold: int | None = None) -> None:
        super().__init__(
            Jaccard(ignore_case, threshold, self._MISSING_BOTH, self._MISSING_EITHER)
        )
