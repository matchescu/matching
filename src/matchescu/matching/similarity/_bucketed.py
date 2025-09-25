from bisect import bisect_left
from collections.abc import Collection
from typing import Union, Mapping, Iterable, Any

from matchescu.matching.similarity._common import Similarity


class BucketedSimilarity(Similarity[float]):
    """Discretize a continuous similarity to a user-defined list of values.

    This similarity wraps another similarity function. The return value of
    the wrapped function is placed in one of many contiguous buckets defined by
    their edges. The bucket edges are supplied by the user.
    The similarity function returns the largest edge smaller than the return
    value or the ``catch_all`` value if the return value is larger than the
    largest bucket edge. This behaviour is the same as setting the largest
    bucket edge to ``float("inf")`` and mapping it to the ``catch_all`` value.

    When the wrapped similarity is expressed as a proportion (i.e. 0 is very
    dissimilar and 1 is identical), this similarity may be initialised with a
    mapping to reverse the semantics or with a catch_all value that is outside
    the [0, 1] interval. When the wrapped similarity expresses a distance (i.e.
    0 means identity and the greater the return value the less similar the
    compared arguments are) then the default arguments work out of the box.
    Note that negative return values are interpreted as ``catch_all`` values
    from the wrapped similarity.
    """

    def __init__(
        self,
        wrapped: Similarity[float],
        buckets: Union[Mapping[float, float], Iterable[float]],
        catch_all: float = 0.0,
        missing_both: float = -1.0,
        missing_either: float = -0.5,
    ) -> None:
        """Initialise a ``BucketedSimilarity`` callable.

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

        assert wrapped is not None, "'wrapped' is missing"
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
        self.__wrapped = wrapped

    @property
    def agreement_levels(self) -> list[float]:
        return self.__values + [self.__catch_all, self._miss_both, self._miss_either]

    def _compute_similarity(self, a: Any, b: Any) -> float:
        ret_val = self.__wrapped(a, b)

        # interpret negative values as catch-all mechanisms
        if ret_val < 0:
            return ret_val

        idx = bisect_left(self.__edges, ret_val)
        # If ret_val > max edge, idx == len(edges),  use catch_all
        return self.__values[idx] if 0 <= idx < len(self.__edges) else self.__catch_all
