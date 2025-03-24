import itertools
from typing import Hashable, Callable

import polars as pl

from matchescu.csg import BinaryComparisonSpaceGenerator
from matchescu.extraction import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._sampling import (
    AttributeComparison,
    PatternEncodedComparison,
)
from matchescu.reference_store.id_table._in_memory import InMemoryIdTable
from matchescu.typing import EntityReference, EntityReferenceIdentifier


class BlockDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        csg: BinaryComparisonSpaceGenerator,
        id_table: InMemoryIdTable,
        left_id: Callable[[EntityReference], Hashable],
        right_id: Callable[[EntityReference], Hashable],
        ground_truth: set[tuple[Hashable, Hashable]],
    ) -> None:
        self.__csg = csg
        self.__id_table = id_table
        self.__lid = left_id
        self.__rid = right_id
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

    def attr_compare(self, config: EntityReferenceComparisonConfig) -> "BlockDataSet":
        self.__sample_factory = AttributeComparison(
            self.__true_matches,
            config,
            self.__lid,
            self.__rid,
            self.__TARGET_COL,
        )
        return self

    def pattern_encoded(
        self, config: EntityReferenceComparisonConfig, possible_outcomes: int = 2
    ) -> "BlockDataSet":
        self.__sample_factory = PatternEncodedComparison(
            self.__true_matches,
            config,
            self.__lid,
            self.__rid,
            self.__TARGET_COL,
            possible_outcomes,
        )
        return self

    def __concat_refs(self, *ref_ids: EntityReferenceIdentifier) -> tuple[tuple, int]:
        refs = list(map(self.__id_table.get, ref_ids))
        return tuple(v for r in refs for v in r), len(refs[0])

    def cross_sources(self) -> "BlockDataSet":
        data = [
            x[0]
            for x in itertools.starmap(
                self.__sample_factory,
                itertools.starmap(self.__concat_refs, self.__csg()),
            )
        ]
        self.__comparison_data = pl.DataFrame(data)
        return self
