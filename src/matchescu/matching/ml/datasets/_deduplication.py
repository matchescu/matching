import itertools
from typing import Any, Iterator, Hashable, Callable

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    EntityReferenceComparisonConfig,
)
from matchescu.matching.ml.datasets._base import BaseDataSet
from matchescu.matching.ml.datasets._reference_comparison import (
    AttributeComparison,
    PatternEncodedComparison,
    NoOp,
)
from matchescu.typing import DataSource, Record, EntityReference


class DeduplicationDataSet(BaseDataSet):
    __TARGET_COL = "y"

    def __init__(
        self,
        source: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
        ref_id: Callable[[EntityReference], Hashable] | None = None,
    ) -> None:
        ref_id = ref_id or (lambda ref: ref[0])
        super().__init__(ground_truth, None, ref_id, ref_id)
        self.__extract = EntityReferenceExtraction(source, ref_id)

    def _comparison_tuples(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        data = list(self.__extract())
        for i, a in enumerate(data):
            for j in range(i + 1, len(data)):
                b = data[j]
                yield a, b
