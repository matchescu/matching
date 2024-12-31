import math
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product
from typing import Callable, Generator, Iterable

import torch

from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
    AttrComparisonSpec,
)
from matchescu.typing import EntityReference


class ComparisonEngine(metaclass=ABCMeta):
    def __init__(self, cmp_config: EntityReferenceComparisonConfig):
        self._config = cmp_config

    @property
    def config(self) -> EntityReferenceComparisonConfig:
        return self._config

    @abstractmethod
    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        pass

    def __call__(self, left_side: EntityReference, right_side: EntityReference) -> dict:
        return self._compare(left_side, right_side)


class NoOp(ComparisonEngine):
    def __init__(self):
        super().__init__(EntityReferenceComparisonConfig())

    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        result = {f"left_{idx}": value for idx, value in enumerate(left, 1)}
        result.update({f"right_{idx}": value for idx, value in enumerate(right, 1)})
        return result


class AttributeComparison(ComparisonEngine):
    @staticmethod
    def __compare_attr_values(
        left_ref: EntityReference,
        right_ref: EntityReference,
        config: AttrComparisonSpec,
    ) -> int:
        a = left_ref[config.left_ref_key]
        b = right_ref[config.right_ref_key]
        return config.match_strategy(a, b)

    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        return {
            spec.label: self.__compare_attr_values(left, right, spec)
            for spec in self._config.specs
        }


class VectorComparison(ComparisonEngine):
    def __init__(
        self,
        cmp_config: EntityReferenceComparisonConfig,
        summarizer: Callable[[Iterable[torch.Tensor]], torch.Tensor] = None,
    ):
        super().__init__(
            cmp_config,
        )
        self._summarizer = summarizer or partial(torch.cat, dim=-1)

    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        comparison_tensors = []
        for spec in self._config.specs:
            left_val = left[spec.left_ref_key]
            right_val = right[spec.right_ref_key]
            if not isinstance(left_val, torch.Tensor) or not isinstance(
                right_val, torch.Tensor
            ):
                continue
            comparison_tensors.append(spec.match_strategy(left_val, right_val))
        summary = self._summarizer(comparison_tensors)
        return {f"col_{idx}": val for idx, val in enumerate(summary, 1)}


class PatternEncodedComparison(ComparisonEngine):
    _BASE = 2

    def __init__(
        self,
        cmp_config: EntityReferenceComparisonConfig,
        possible_outcomes: int = 2,
    ):
        super().__init__(cmp_config)
        self._possible_outcomes = possible_outcomes

    def _generate_binary_patterns(self) -> Generator[tuple, None, None]:
        possible_outcomes = tuple(range(self._possible_outcomes))
        yield from product(possible_outcomes, repeat=len(self._config))

    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
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
