from abc import abstractmethod
from typing import Any, Type, Iterable

from matchescu.matching.attribute import (
    BinarySimilarityMatchOnThreshold,
    RawMatch,
    SimilarityMatch,
    TernarySimilarityMatchOnThreshold,
)
from matchescu.matching.entity_reference._attr_spec import AttrComparisonSpec
from matchescu.matching.similarity import (
    BoundedNumericDifferenceSimilarity,
    ExactMatch,
    Jaro,
    Jaccard,
    JaroWinkler,
    LevenshteinSimilarity,
    Similarity,
    LevenshteinDistance,
    Norm,
)


class EntityReferenceComparisonConfig:
    def __init__(self):
        self.__specs = []

    @classmethod
    @abstractmethod
    def _new_similarity_threshold_match_strategy(
        cls, similarity: Similarity, *args
    ) -> SimilarityMatch:
        pass

    @classmethod
    def _new_spec(
        cls,
        similarity_type: Type,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float,
        *args: Any
    ) -> AttrComparisonSpec:
        return AttrComparisonSpec(
            label=label,
            left_ref_key=left_key,
            right_ref_key=right_key,
            match_strategy=cls._new_similarity_threshold_match_strategy(
                similarity_type(*args), threshold
            ),
        )

    def exact(
        self, label: str, left_key: int | str, right_key: int | str
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(self._new_spec(ExactMatch, label, left_key, right_key, 1))
        return self

    def diff(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                Norm,
                label,
                left_key,
                right_key,
                0.0,
            )
        )
        return self

    def bounded_diff(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float = 0.5,
        max_diff: float = 1.0,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                BoundedNumericDifferenceSimilarity,
                label,
                left_key,
                right_key,
                threshold,
                max_diff,
            )
        )
        return self

    def jaro(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(Jaro, label, left_key, right_key, threshold, ignore_case)
        )
        return self

    def jaccard(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float = 0.5,
        gram_size: int | None = None,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                Jaccard, label, left_key, right_key, threshold, ignore_case, gram_size
            )
        )
        return self

    def jaro_winkler(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                JaroWinkler, label, left_key, right_key, threshold, ignore_case
            )
        )
        return self

    def levenshtein(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                LevenshteinSimilarity,
                label,
                left_key,
                right_key,
                threshold,
                ignore_case,
            )
        )
        return self

    def levenshtein_distance(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                LevenshteinDistance,
                label,
                left_key,
                right_key,
                threshold,
                ignore_case,
            )
        )
        return self

    def __iter__(self):
        return self.__specs

    def __len__(self):
        return len(self.__specs)

    @property
    def specs(self) -> Iterable[AttrComparisonSpec]:
        return self.__specs


class FellegiSunterComparison(EntityReferenceComparisonConfig):
    @classmethod
    def _new_similarity_threshold_match_strategy(
        cls, similarity: Similarity, *args
    ) -> SimilarityMatch:
        return TernarySimilarityMatchOnThreshold(similarity, *args)


class NaiveBayesComparison(EntityReferenceComparisonConfig):
    @classmethod
    def _new_similarity_threshold_match_strategy(
        cls, similarity: Similarity, *args
    ) -> SimilarityMatch:
        return BinarySimilarityMatchOnThreshold(similarity, *args)


class RawComparison(EntityReferenceComparisonConfig):
    @classmethod
    def _new_similarity_threshold_match_strategy(
        cls, similarity: Similarity, *_
    ) -> SimilarityMatch:
        return RawMatch(similarity)
