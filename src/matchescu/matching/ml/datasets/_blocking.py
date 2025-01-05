from typing import Hashable, Callable, Iterator

from matchescu.matching.blocking import BlockEngine
from matchescu.matching.ml.datasets._base import BaseDataSet
from matchescu.typing import EntityReference


class BlockDataSet(BaseDataSet):
    def __init__(
        self,
        block_engine: BlockEngine,
        ground_truth: set[tuple[Hashable, Hashable]],
        left_id: Callable[[EntityReference], Hashable] | None = None,
        right_id: Callable[[EntityReference], Hashable] | None = None,
    ) -> None:
        left_id = left_id or (lambda x: x[0])
        right_id = right_id or (lambda x: x[0])
        super().__init__(ground_truth, None, left_id, right_id)
        self.__engine = block_engine

    def _comparison_tuples(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        yield from self.__engine.candidate_pairs()
