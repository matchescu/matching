from abc import abstractmethod, ABCMeta
from typing import Any, Optional, TypeVar, Generic

from matchescu.matching.attribute._match_result import (
    TernaryResult,
    BinaryResult,
)
from matchescu.matching.similarity._common import Similarity

TResult = TypeVar("TResult", contravariant=True)


class SimilarityMatch(Generic[TResult]):
    def __init__(self, similarity: Similarity) -> None:
        self.__compute_similarity = similarity
        self.__similarity: float | None = None

    @abstractmethod
    def _to_match_result(self) -> TResult:
        """Express the computed similarity as a ``MatchResult``.

        Use the ``self.similarity`` property to access the computed similarity.
        """
        pass

    @abstractmethod
    def _handle_missing_data(self, a: Any, b: Any) -> Optional[TResult]:
        """Provide a match result for missing data.

        If not applicable, simply return ``None``.
        """

    def __call__(self, a: Any, b: Any) -> TResult:
        if (result := self._handle_missing_data(a, b)) is not None:
            return result

        self.__similarity = self.__compute_similarity(a, b)
        return self._to_match_result()

    @property
    def similarity(self) -> float | None:
        return self.__similarity


class BinarySimilarityMatch(SimilarityMatch[BinaryResult], metaclass=ABCMeta):
    """Match strategy where the result can only be one of only 2 values.

    This strategy is not as descriptive as the ``TernarySimilarityMatch``, but
    it is useful for approaches similar to the Naive Bayes model which can only
    work with binary valued features.
    """

    def _handle_missing_data(self, a: Any, b: Any) -> Optional[BinaryResult]:
        if a is None and b is None:
            return BinaryResult.Negative
        return None


class TernarySimilarityMatch(SimilarityMatch[TernaryResult], metaclass=ABCMeta):
    """Match strategy where the result can be one of 3 possible values.

    This matching strategy always returns ``TernaryResult`` instances, allowing
    it to handle the case when there's missing data.
    """

    def _handle_missing_data(self, a: Any, b: Any) -> Optional[TResult]:
        if a is None and b is None:
            return TernaryResult.NoComparisonData
        return None


class _ThresholdSimilarityMixin(
    SimilarityMatch[TResult], Generic[TResult], metaclass=ABCMeta
):
    def __init__(
        self,
        similarity: Similarity,
        match: TResult,
        non_match: TResult,
        threshold: float = 0.5,
    ):
        super().__init__(similarity)
        self.__threshold = threshold
        self.__match = match
        self.__non_match = non_match

    def _to_match_result(self) -> TResult:
        return self.__match if self.similarity >= self.__threshold else self.__non_match


class TernarySimilarityMatchOnThreshold(
    TernarySimilarityMatch, _ThresholdSimilarityMixin
):
    """Match strategy returning ternary results that uses thresholds.

    The strategy here is to assume values match when their similarity score is
    above a given threshold value. Otherwise, assume that values don't match.
    The handling of insufficient data (such as when both values are ``None``) is
    handled likewise to the other ``TernarySimilarityMatch`` strategies.
    """

    def __init__(self, similarity: Similarity, threshold: float = 0.5) -> None:
        super().__init__(
            similarity, TernaryResult.Match, TernaryResult.NonMatch, threshold
        )


class BinarySimilarityMatchOnThreshold(
    BinarySimilarityMatch, _ThresholdSimilarityMixin
):
    """Match strategy returning binary results that uses thresholds.

    The strategy here is to assume values match when their similarity score is
    above a given threshold value. Otherwise, assume that values don't match.
    The handling of insufficient data (such as when both values are ``None``) is
    handled likewise to the other ``BinarySimilarityMatch`` strategies.
    """

    def __init__(self, similarity: Similarity, threshold: float = 0.5) -> None:
        super().__init__(
            similarity, BinaryResult.Positive, BinaryResult.Negative, threshold
        )


class RawMatch(SimilarityMatch[float]):
    def _to_match_result(self) -> float:
        return self.similarity

    def _handle_missing_data(self, a: Any, b: Any) -> Optional[float]:
        return None  # defer handling to the similarity function
