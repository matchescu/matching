import itertools
from abc import ABCMeta, abstractmethod
from typing import Hashable, Callable, Iterator

import numpy as np
import polars as pl

from matchescu.matching.entity_reference import EntityReferenceComparisonConfig
from matchescu.matching.ml.datasets._reference_comparison import (
    AttributeComparison,
    PatternEncodedComparison,
    VectorComparison,
    NoOp,
)
from matchescu.typing import EntityReference


class BaseDataSet(metaclass=ABCMeta):
    __TARGET_COL = "y"

    def __init__(
        self,
        ground_truth: set[tuple[Hashable, Hashable]],
        sample_factory: (
            Callable[[EntityReference, EntityReference], dict] | None
        ) = None,
        left_id: Callable[[EntityReference], Hashable] | None = None,
        right_id: Callable[[EntityReference], Hashable] | None = None,
    ) -> None:
        self.__true_matches = ground_truth
        self.__sample_factory = sample_factory or NoOp()
        self.__left_id = left_id
        self.__right_id = right_id

        self.__comparison_data: pl.DataFrame | None = None
        self.__columns: set[str] | None = None

    @property
    def target_vector(self) -> np.ndarray:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        if self.__TARGET_COL not in self.__columns:
            raise ValueError("target vector was not computed")
        return self.__comparison_data[self.__TARGET_COL].to_numpy()

    @property
    def feature_matrix(self) -> np.ndarray:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return (
            self.__comparison_data.drop([self.__TARGET_COL]).to_numpy()
            if self.__TARGET_COL in self.__columns
            else self.__comparison_data.to_numpy()
        )

    @property
    def training_data(self) -> np.ndarray:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return self.__comparison_data.to_numpy()

    def attr_compare(self, config: EntityReferenceComparisonConfig) -> "BaseDataSet":
        self.__sample_factory = AttributeComparison(config)
        return self

    def pattern_encoded(
        self, config: EntityReferenceComparisonConfig, possible_outcomes: int = 2
    ) -> "BaseDataSet":
        self.__sample_factory = PatternEncodedComparison(config, possible_outcomes)
        return self

    def vector_compare(self, config: EntityReferenceComparisonConfig) -> "BaseDataSet":
        self.__sample_factory = VectorComparison(config)
        return self

    @abstractmethod
    def _comparison_tuples(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        yield from ()

    def _compare_references(self, a: EntityReference, b: EntityReference) -> dict:
        comparison_result = self.__sample_factory(a, b)
        if self.__left_id and self.__right_id:
            compared_ids = (self.__left_id(a), self.__right_id(b))
            comparison_result[self.__TARGET_COL] = compared_ids in self.__true_matches
        return comparison_result

    def cross_sources(self) -> "BaseDataSet":
        self.__comparison_data = pl.DataFrame(
            itertools.starmap(self._compare_references, self._comparison_tuples())
        )
        self.__columns = set(self.__comparison_data.columns)
        return self
