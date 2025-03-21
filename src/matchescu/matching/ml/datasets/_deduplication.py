from functools import partial
from typing import Any

import polars as pl

from matchescu.extraction import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._sampling import (
    AttributeComparison,
    PatternEncodedComparison,
)


class DeduplicationDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        extractor: EntityReferenceExtraction,
        ground_truth: set[tuple[Any, Any]],
    ) -> None:
        self.__extract = extractor
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

    def cross_sources(self) -> "DeduplicationDataSet":
        if self.__sample_factory is None:
            raise ValueError("specify type of sampling")
        source = list(self.__extract())
        if len(source) < 1:
            raise ValueError("no data")

        mid = len(source[0])
        data = []
        for i, left_row in enumerate(source):
            for j in range(i + 1, len(source)):
                row = {
                    f"left_column_{idx+1}": value for idx, value in enumerate(left_row)
                }
                row.update(
                    {
                        f"right_column_{idx + 1}": value
                        for idx, value in enumerate(source[j])
                    }
                )
                data.append(row)
        sample_factory = partial(self.__sample_factory, divider=mid)
        df = pl.DataFrame(data)
        self.__comparison_data = df.map_rows(sample_factory).unnest("column_0")
        return self
