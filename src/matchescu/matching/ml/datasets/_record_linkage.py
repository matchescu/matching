from functools import partial
from typing import Any

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._sampling import (
    AttributeComparison,
    PatternEncodedComparison,
)
from matchescu.typing import DataSource, Record


class RecordLinkageDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        left: DataSource[Record],
        right: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
    ) -> None:
        self.__extract_left = EntityReferenceExtraction(left, lambda ref: ref[0])
        self.__extract_right = EntityReferenceExtraction(right, lambda ref: ref[0])
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
    ) -> "RecordLinkageDataSet":
        self.__sample_factory = AttributeComparison(
            self.__true_matches,
            config,
            self.__extract_left.identify,
            self.__extract_right.identify,
            self.__TARGET_COL,
        )
        return self

    def pattern_encoded(
        self, config: EntityReferenceComparisonConfig, possible_outcomes: int = 2
    ) -> "RecordLinkageDataSet":
        self.__sample_factory = PatternEncodedComparison(
            self.__true_matches,
            config,
            self.__extract_left.identify,
            self.__extract_right.identify,
            self.__TARGET_COL,
            possible_outcomes,
        )
        return self

    def cross_sources(self) -> "RecordLinkageDataSet":
        if self.__sample_factory is None:
            raise ValueError("specify type of sampling")
        left = self.__with_col_suffix(self.__extract_left, "_left")
        right = self.__with_col_suffix(self.__extract_right, "_right")
        cross_join = left.join(right, how="cross")
        sample_factory = partial(self.__sample_factory, divider=len(left.columns))
        self.__comparison_data = cross_join.map_rows(sample_factory).unnest("column_0")
        return self
