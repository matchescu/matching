import itertools
import re
from functools import partial
from typing import Iterator


from matchescu.data import EntityReferenceExtraction
from matchescu.matching.blocking._block import Block
from matchescu.typing import EntityReference


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


def _process_candidate(source: str, candidate: EntityReference, block: Block, jaccard_threshold: float, column: str, center_lemmas: set[str]) -> Iterator[tuple]:
    cand_lemmas = _tokens(candidate[column])
    similarity = _jaccard_coefficient(center_lemmas, cand_lemmas)
    if similarity >= jaccard_threshold:
        block.add_reference(source, candidate)
        yield source, candidate


def _canopy_clustering(all_data: list[tuple[str, EntityReference]], column: int, jaccard_threshold: float = 0.5) -> Iterator[Block]:
    while len(all_data) > 0:
        ref_source, ref_data = all_data.pop(0)
        ref_col_value = ref_data[column]
        reference_tokens = _tokens(ref_col_value)
        block = Block(key=f"{ref_source}-{ref_col_value}").add_reference(ref_source, ref_data)
        process_candidate = partial(_process_candidate, block=block, jaccard_threshold=jaccard_threshold, column=column, center_lemmas=reference_tokens)

        for to_remove in itertools.starmap(process_candidate, all_data):
            for item in to_remove:
                all_data.remove(item)

        yield block


class BlockingEngine:
    def __init__(self, reference_extractors: list[EntityReferenceExtraction]):
        self._blocks = []
        self._all_data = [(x.source_name, r) for x in reference_extractors for r in x()]

    def canopy_clustering(self, column: int, threshold: float = 0.5) -> "BlockingEngine":
        self._blocks.extend(
            _canopy_clustering(self._all_data.copy(), column, threshold)
        )
        return self

    @staticmethod
    def _at_least_two_sources(block: Block) -> bool:
        return len(block.references) > 1

    def cross_sources_filter(self) -> "BlockingEngine":
        self._blocks = list(filter(self._at_least_two_sources, self._blocks))
        return self

    @property
    def blocks(self) -> list[Block]:
        return self._blocks