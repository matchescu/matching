import re
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Hashable, Callable

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.blocking._block import Block
from matchescu.matching.extraction import ListDataSource
from matchescu.typing import EntityReference, DataSource, Record

token_regexp = re.compile(r"[\d\W_]+")


def _clean(tok: str) -> str:
    if tok is None:
        return ""
    return tok.strip("\t\n\r\a ").lower()


def _tokens(text: str) -> set[str]:
    if text is None:
        return set()
    return set(t for t in map(_clean, token_regexp.split(text)) if t)


def _jaccard_coefficient(a: set, b: set) -> float:
    if a is None or b is None:
        return 0
    return len(a.intersection(b)) / len(a.union(b))


@dataclass
class BlockEngineItem:
    source: str
    reference: EntityReference
    ref_id: Callable[[EntityReference], Hashable]


def _process_candidate(
    item: BlockEngineItem,
    block: Block,
    jaccard_threshold: float,
    column: str,
    center_lemmas: set[str],
) -> Iterator[BlockEngineItem]:
    cand_lemmas = _tokens(item.reference[column])
    similarity = _jaccard_coefficient(center_lemmas, cand_lemmas)
    if similarity >= jaccard_threshold:
        block.append(item.ref_id(item.reference), source_name=item.source)
        yield item


def _canopy_clustering(
    all_data: list[BlockEngineItem], column: int, jaccard_threshold: float = 0.5
) -> Iterator[Block]:
    while len(all_data) > 0:
        item = all_data.pop(0)
        ref_col_value = item.reference[column]
        reference_tokens = _tokens(ref_col_value)
        block_key = f"{item.source}-{'-'.join(reference_tokens)}"
        block = Block(key=block_key).append(
            item.ref_id(item.reference), source_name=item.source
        )
        process_candidate = partial(
            _process_candidate,
            block=block,
            jaccard_threshold=jaccard_threshold,
            column=column,
            center_lemmas=reference_tokens,
        )

        for to_remove in map(process_candidate, all_data):
            for item in to_remove:
                all_data.remove(item)

        yield block


@dataclass
class BlockingMetrics:
    pair_completeness: float
    pair_quality: float
    reduction_ratio: float


class BlockEngine:
    def __init__(self):
        self._blocks: list[Block] = []
        self._all_data: dict[Hashable, BlockEngineItem] = {}
        self._candidates: list[tuple[BlockEngineItem, BlockEngineItem]] = []
        self._total_possible_pairs: int = 0

    def _update_candidate_pairs(self) -> None:
        self._candidates.clear()
        for block in self._blocks:
            for ref_id1, ref_id2 in block.candidate_pairs():
                self._candidates.append(
                    (self._all_data[ref_id1], self._all_data[ref_id2])
                )

    def add_source(
        self,
        data_source: DataSource[Record],
        id_factory: Callable[[EntityReference], Hashable],
    ) -> "BlockEngine":
        n_records = len(data_source)
        if n_records == 0:
            return self

        extract_references = EntityReferenceExtraction(data_source, id_factory)
        self._total_possible_pairs = (
            n_records
            if self._total_possible_pairs == 0
            else self._total_possible_pairs * n_records
        )
        self._all_data.update(
            {
                extract_references.identify(ref): BlockEngineItem(
                    data_source.name, ref, extract_references.identify
                )
                for ref in extract_references()
            }
        )
        return self

    def jaccard_canopy(
        self, column: int, jaccard_threshold: float = 0.5
    ) -> "BlockEngine":
        self._blocks.extend(
            _canopy_clustering(list(self._all_data.values()), column, jaccard_threshold)
        )
        self._update_candidate_pairs()
        return self

    @staticmethod
    def _at_least_two_sources(block: Block) -> bool:
        return len(block.references) > 1

    def only_multi_source_blocks(self) -> "BlockEngine":
        self._blocks = list(filter(self._at_least_two_sources, self._blocks))
        return self

    @property
    def blocks(self) -> list[Block]:
        return self._blocks

    def candidate_pairs(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        yield from (
            map(lambda x: x.reference, candidate) for candidate in self._candidates
        )

    def calculate_metrics(
        self, ground_truth: set[tuple[Hashable, Hashable]]
    ) -> BlockingMetrics:
        candidate_ids = set(
            tuple(map(lambda x: x.ref_id(x.reference), candidate))
            for candidate in self._candidates
        )
        true_positive_pairs = ground_truth.intersection(candidate_ids)
        pc = len(true_positive_pairs) / len(ground_truth) if ground_truth else 0
        pq = len(true_positive_pairs) / len(candidate_ids) if candidate_ids else 0
        rr = 1 - (
            len(candidate_ids) / self._total_possible_pairs
            if self._total_possible_pairs
            else 0
        )
        return BlockingMetrics(pc, pq, rr)

    def list_source_names(self) -> Iterator[str]:
        visited = set()
        for block in self._blocks:
            for source_name in block.references.keys():
                if source_name not in visited:
                    yield source_name
                    visited.add(source_name)
