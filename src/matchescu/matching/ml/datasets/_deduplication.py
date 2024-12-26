import itertools
from typing import Any, Iterator

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._reference_comparison import (
    AttributeComparison,
    PatternEncodedComparison, NoOp,
)
from matchescu.typing import DataSource, Record


class DeduplicationDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        source: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
    ) -> None:
        self.__extract = EntityReferenceExtraction(source, lambda ref: ref[0])
        self.__true_matches = ground_truth
        self.__comparison_data = None
        self.__sample_factory = None

    @property
    def target_vector(self) -> pl.DataFrame:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return self.__comparison_data[self.__TARGET_COL]

    @property
    def feature_matrix(self) -> pl.DataFrame:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return self.__comparison_data.drop([self.__TARGET_COL])

    @staticmethod
    def __with_col_suffix(
        extract: EntityReferenceExtraction, suffix: str
    ) -> pl.DataFrame:
        df = pl.DataFrame(extract())
        return df.rename({key: f"{key}{suffix}" for key in df.columns})

    def attr_compare(
        self, config: EntityReferenceComparisonConfig
    ) -> "DeduplicationDataSet":
        self.__sample_factory = AttributeComparison(
            self.__true_matches,
            config,
            self.__extract.identify,
            self.__extract.identify,
            self.__TARGET_COL,
        )
        return self

    def pattern_encoded(
        self, config: EntityReferenceComparisonConfig, possible_outcomes: int = 2
    ) -> "DeduplicationDataSet":
        self.__sample_factory = PatternEncodedComparison(
            self.__true_matches,
            config,
            self.__extract.identify,
            self.__extract.identify,
            self.__TARGET_COL,
            possible_outcomes,
        )
        return self

    @staticmethod
    def _self_product(data: list) -> Iterator[tuple]:
        for i, a in enumerate(data):
            for j in range(i+1, len(data)):
                b = data[j]
                yield a, b

    def cross_sources(self) -> "DeduplicationDataSet":
        if self.__sample_factory is None:
            raise ValueError("specify type of sampling")
        source = list(self.__extract())

        if len(source) < 1:
            raise ValueError("no data")
        no_op = NoOp(self.__true_matches, EntityReferenceComparisonConfig(), self.__extract.identify, self.__extract.identify, self.__TARGET_COL)
        sample_factory = self.__sample_factory or no_op

        self.__comparison_data = pl.DataFrame(itertools.starmap(sample_factory, self._self_product(source)))
        return self
