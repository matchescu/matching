import math
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product
from typing import Any, Callable, Hashable, Generator

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    AttrComparisonSpec,
    EntityReferenceComparisonConfig,
)
from matchescu.typing import DataSource, Record, EntityReference


class Sampling(metaclass=ABCMeta):
    def __init__(
        self,
        ground_truth: set[tuple[Hashable, Hashable]],
        cmp_config: EntityReferenceComparisonConfig,
        left_id: Callable[[EntityReference], Hashable],
        right_id: Callable[[EntityReference], Hashable],
        target_col_name: str,
    ):
        self._gt = ground_truth
        self._config = cmp_config
        self._left_id = left_id
        self._right_id = right_id
        self._target_col = target_col_name

    @abstractmethod
    def _process_cross_join_record(
        self, left: EntityReference, right: EntityReference
    ) -> dict:
        pass

    def __call__(self, cross_join_row: tuple, divider: int) -> tuple[dict]:
        left_side = cross_join_row[:divider]
        right_side = cross_join_row[divider:]
        result = self._process_cross_join_record(left_side, right_side)
        result[self._target_col] = int(
            (self._left_id(left_side), self._right_id(right_side)) in self._gt
        )
        return (result,)  # need to return a tuple


class AttributeComparison(Sampling):
    @staticmethod
    def __compare_attr_values(
        left_ref: EntityReference,
        right_ref: EntityReference,
        config: AttrComparisonSpec,
    ) -> int:
        a = left_ref[config.left_ref_key]
        b = right_ref[config.right_ref_key]
        return config.match_strategy(a, b)

    def _process_cross_join_record(
        self, left: EntityReference, right: EntityReference
    ) -> dict:
        return {
            spec.label: self.__compare_attr_values(left, right, spec)
            for spec in self._config.specs
        }


class PatternEncodedComparison(Sampling):
    _BASE = 2

    def __init__(
        self,
        ground_truth: set[tuple[Hashable, Hashable]],
        cmp_config: EntityReferenceComparisonConfig,
        left_id: Callable[[EntityReference], Hashable],
        right_id: Callable[[EntityReference], Hashable],
        target_col_name: str,
        possible_outcomes: int = 2,
    ):
        super().__init__(ground_truth, cmp_config, left_id, right_id, target_col_name)
        self._possible_outcomes = possible_outcomes

    def _generate_binary_patterns(self) -> Generator[tuple, None, None]:
        possible_outcomes = tuple(range(self._possible_outcomes))
        yield from product(possible_outcomes, repeat=len(self._config))

    def _process_cross_join_record(
        self, left: EntityReference, right: EntityReference
    ) -> dict:
        comparison_results = [
            spec.match_strategy(left[spec.left_ref_key], right[spec.right_ref_key])
            for spec in self._config.specs
        ]
        sample = {}
        for pattern in self._generate_binary_patterns():
            pattern_value = 0
            for idx, current in enumerate(zip(pattern, comparison_results)):
                expectation, actual = current
                coefficient = math.pow(self._BASE, idx)
                pattern_value += expectation * actual * coefficient
            sample["".join(map(str, pattern))] = pattern_value
        return sample


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
