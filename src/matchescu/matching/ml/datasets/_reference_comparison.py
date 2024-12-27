import math
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product
from typing import Hashable, Callable, Generator, Iterable

import torch

from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
    AttrComparisonSpec,
)
from matchescu.typing import EntityReference


class ComparisonEngine(metaclass=ABCMeta):
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
    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        pass

    def __call__(self, left_side: EntityReference, right_side: EntityReference) -> dict:
        result = self._compare(left_side, right_side)
        lid = self._left_id(left_side)
        rid = self._right_id(right_side)
        result[self._target_col] = int((lid, rid) in self._gt)
        return result


class NoOp(ComparisonEngine):
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
        ground_truth: set[tuple[Hashable, Hashable]],
        cmp_config: EntityReferenceComparisonConfig,
        left_id: Callable[[EntityReference], Hashable],
        right_id: Callable[[EntityReference], Hashable],
        target_col_name: str,
        tensor_summarizer: Callable[[Iterable[torch.Tensor]], torch.Tensor] = None,
    ):
        super().__init__(ground_truth, cmp_config, left_id, right_id, target_col_name)
        self._tensor_summarizer = tensor_summarizer or partial(torch.cat, dim=-1)

    def _compare(self, left: EntityReference, right: EntityReference) -> dict:
        comparison_tensors = []
        for spec in self._config.specs:
            lval = left[spec.left_ref_key]
            rval = right[spec.right_ref_key]
            if not isinstance(lval, torch.Tensor) or not isinstance(rval, torch.Tensor):
                continue
            comparison_tensors.append(spec.match_strategy(lval, rval))
        summary = self._tensor_summarizer(comparison_tensors)
        return {f"col_{idx}": val for idx, val in enumerate(summary, 1)}


class PatternEncodedComparison(ComparisonEngine):
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
