import itertools
from abc import ABCMeta, abstractmethod

from matchescu.matching.blocking._tokenization import tokenize_reference
from matchescu.reference_store.id_table._in_memory import InMemoryIdTable
from matchescu.typing import EntityReferenceIdentifier, EntityReference


class Filter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *ref_ids: EntityReferenceIdentifier) -> bool:
        pass


class MultiSourceFilter(Filter):
    def __call__(self, *ref_ids: EntityReferenceIdentifier) -> bool:
        sources = set(ref_id.source for ref_id in ref_ids)
        return len(sources) > 1


class JaccardSimilarityFilter(Filter):
    def __init__(self, id_table: InMemoryIdTable, min_similarity: float = 0.5) -> None:
        self._id_table = id_table
        self._min_similarity = min_similarity

    @staticmethod
    def _similarity(ref1: EntityReference, ref2: EntityReference) -> float:
        tok1 = set(tokenize_reference(ref1))
        tok2 = set(tokenize_reference(ref2))
        n_common = len(tok1 & tok2)
        n_total = len(tok1 | tok2)
        return n_common / n_total if n_total != 0 else 0

    def __call__(self, *ref_ids: EntityReferenceIdentifier) -> bool:
        refs = [self._id_table.get(ref_id.source, ref_id.label) for ref_id in ref_ids]
        scores = [self._similarity(*comb) for comb in itertools.combinations(refs, 2)]
        return max(scores) >= self._min_similarity
