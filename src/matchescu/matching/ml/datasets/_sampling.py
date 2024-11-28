import math
from abc import ABCMeta, abstractmethod
from itertools import product
from typing import Hashable, Callable, Generator

from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
    AttrComparisonSpec,
)
from matchescu.typing import EntityReference


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
