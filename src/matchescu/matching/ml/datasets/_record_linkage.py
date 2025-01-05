import itertools
from typing import Callable, Hashable, Iterator

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.ml.datasets._base import BaseDataSet
from matchescu.typing import EntityReference, DataSource, Record


class RecordLinkageDataSet(BaseDataSet):
    def __init__(
        self,
        left: DataSource[Record],
        right: DataSource[Record],
        gt: set[tuple[Hashable, Hashable]],
        left_id: Callable[[EntityReference], Hashable] | None = None,
        right_id: Callable[[EntityReference], Hashable] | None = None,
    ) -> None:
        left_id = left_id or (lambda x: x[0])
        right_id = right_id or (lambda x: x[0])
        super().__init__(gt, None, left_id, right_id)
        self.__extract_left = EntityReferenceExtraction(left, left_id)
        self.__extract_right = EntityReferenceExtraction(right, right_id)

    def _comparison_tuples(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        yield from itertools.product(self.__extract_left(), self.__extract_right())
