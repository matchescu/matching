from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from decimal import Decimal
from typing import Any, Iterable, Mapping, Union
from bisect import bisect_left

import numpy as np

from matchescu.matching.similarity import Similarity


class NumericSimilarity(Similarity[float], metaclass=ABCMeta):
    __SUPPORTED_TYPES = (int, float, Decimal)

    def __init__(self, missing_both: float = -1.0, missing_either: float = 0.0):
        super().__init__(missing_both, missing_either)
        self.__unsupported_type = missing_either

    @abstractmethod
    def _compute_float_similarity(self, a: float, b: float) -> float:
        raise NotImplementedError()

    def _compute_similarity(self, a: Any, b: Any) -> float:
        if not isinstance(a, self.__SUPPORTED_TYPES) or not isinstance(
            b, self.__SUPPORTED_TYPES
        ):
            return self.__unsupported_type
        return self._compute_float_similarity(float(a), float(b))


class Norm(NumericSimilarity):
    def _compute_float_similarity(self, a: float, b: float) -> float:
        return abs(float(a - b))


class BoundedNumericDifferenceSimilarity(NumericSimilarity):
    def __init__(self, max_diff: float = 1.0) -> None:
        super().__init__()
        self.__max_diff = max_diff
        # 11 bins including both endpoints: 0.0, 0.1, ..., 1.0
        self.__steps = np.linspace(0.0, 1.0, 11).tolist()

    def _compute_float_similarity(self, a: float, b: float) -> float:
        diff = float(abs(a - b))
        diff = float(np.clip(diff, 0.0, self.__max_diff))
        # Proximity in [0, 1]: 1.0 means identical, 0.0 means at/beyond max_diff
        proximity = 1.0 - (diff / self.__max_diff)
        n = len(self.__steps)
        idx = int(round(proximity * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return self.__steps[idx]


class BucketedNorm(NumericSimilarity):
    """Absolute-difference similarity with user-defined edges.

    This similarity computes the absolute difference between two numbers. Then
    the resulting value is placed in a bucket. This calls for comparing the
    value with the edges of the buckets passed as input by the user. The largest
    edge smaller than the norm is selected. The similarity function returns
    either that edge, ``catch_all`` (if the norm is larger than the max
    configured bucket). This is the same as setting the largest bucket edge to
    ``float("inf")`` and mapping it to the ``catch_all`` value.
    """

    def __init__(
        self,
        buckets: Union[Mapping[float, float], Iterable[float]],
        catch_all: float = 0.0,
        missing_both: float = -1.0,
        missing_either: float = -0.5,
    ) -> None:
        """Initialise a ``BucketedNorm`` similarity function.

        :param buckets: a ``Mapping[float, float]`` or a 'real' collection
            (``list``, ``set``, ``tuple``, ...). If the argument is a mapping, the
            keys are the user-defined edges and the values are what the similarity
            function should return. If the argument is a list, the edges and return
            values are equal.
        :param catch_all: a value that should be returned if the norm falls outside
            any of the ``buckets``.
        :param missing_both: the return value when both similarity input arguments
            are ``None``
        :param missing_either: the return value when one of the similarity input
            arguments is ``None``
        :raises: ``AssertionError`` if ``buckets`` is not of a supported type or is
            empty.
        """
        super().__init__(missing_both, missing_either)
        assert isinstance(buckets, Mapping) or (
            isinstance(buckets, Collection)
            and not isinstance(buckets, (str, bytes, bytearray, memoryview))
        ), f"unsupported 'buckets' type: '{type(buckets).__name__}'"
        assert len(buckets) > 0, "'buckets' is empty"

        rules = (
            {float(k): float(v) for k, v in buckets.items()}
            if isinstance(buckets, Mapping)
            else {float(x): float(x) for x in buckets}
        )
        # Sort edges ascending; if duplicates exist, the last value wins via dict semantics
        self.__edges = sorted(rules.keys())
        self.__values = [rules[e] for e in self.__edges]
        self.__catch_all = float(catch_all)

    def _compute_float_similarity(self, a: float, b: float) -> float:
        d = abs(float(a) - float(b))
        idx = bisect_left(self.__edges, d)

        # If d > max edge, idx == len(edges),  use catch_all
        return self.__values[idx] if 0 <= idx < len(self.__edges) else self.__catch_all
