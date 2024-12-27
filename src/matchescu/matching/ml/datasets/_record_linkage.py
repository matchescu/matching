import itertools
from typing import Any, Callable, Hashable

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._reference_comparison import (
    AttributeComparison,
    NoOp,
    PatternEncodedComparison,
    VectorComparison,
)
from matchescu.matching.ml.datasets._torch import PlTorchDataset
from matchescu.typing import DataSource, Record, EntityReference


class RecordLinkageDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        left: DataSource[Record],
        right: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
        left_id: Callable[[EntityReference], Hashable] = None,
        right_id: Callable[[EntityReference], Hashable] = None,
    ) -> None:
        left_id = left_id or (lambda ref: ref[0])
        right_id = right_id or (lambda ref: ref[0])
        self.__extract_left = EntityReferenceExtraction(left, left_id)
        self.__extract_right = EntityReferenceExtraction(right, right_id)
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

    def create_torch_dataset(self):
        return PlTorchDataset(self.__comparison_data, self.__TARGET_COL)

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

    def vector_compare(
        self, config: EntityReferenceComparisonConfig
    ) -> "RecordLinkageDataSet":
        self.__sample_factory = VectorComparison(
            self.__true_matches,
            config,
            self.__extract_left.identify,
            self.__extract_right.identify,
            self.__TARGET_COL,
        )
        return self

    def cross_sources(self) -> "RecordLinkageDataSet":
        left_entity_references = list(self.__extract_left())
        right_entity_references = list(self.__extract_right())
        cross_entity_references = itertools.product(
            left_entity_references, right_entity_references
        )

        no_op = NoOp(
            self.__true_matches,
            EntityReferenceComparisonConfig(),
            self.__extract_left.identify,
            self.__extract_right.identify,
            self.__TARGET_COL,
        )
        sample_factory = self.__sample_factory or no_op

        self.__comparison_data = pl.DataFrame(
            itertools.starmap(sample_factory, cross_entity_references)
        )
        return self
