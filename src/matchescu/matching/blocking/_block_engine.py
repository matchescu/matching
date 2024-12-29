import re
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Hashable, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import clean as remove_stopwords

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.blocking._block import Block
from matchescu.typing import EntityReference, DataSource, Record

token_regexp = re.compile(r"[\d\W_]+")


def _clean(tok: str) -> str:
    if tok is None:
        return ""
    return tok.strip("\t\n\r\a ").lower()


def _tokens(text: str, language: str = "en", min_length: int = 3) -> set[str]:
    if text is None:
        return set()
    return set(
        t for t in remove_stopwords(
            list(map(_clean, token_regexp.split(text))),
            language
        ) if t and len(t) >= min_length
    )


def _jaccard_coefficient(a: set, b: set) -> float:
    if a is None or b is None:
        return 0
    return len(a.intersection(b)) / len(a.union(b))


@dataclass
class BlockEngineItem:
    source: str
    reference: EntityReference
    ref_id: Callable[[EntityReference], Hashable]

    def __repr__(self):
        return f"{{src={self.source},id={self.ref_id(self.reference)}}}"

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
        return self

    def tf_idf(self, column: int) -> "BlockEngine":
        ref_ids = []
        corpus = []
        for ref_id, item in self._all_data.items():
            corpus.append(" ".join(tok for tok in _tokens(str(item.reference[column]))))
            ref_ids.append(ref_id)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        token_inverted_map = vectorizer.get_feature_names_out()

        blocks: dict[str, Block] = {}
        for idx, ref_id in enumerate(ref_ids):
            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            highest_score_idx = np.argmax(tfidf_scores)
            highest_score_token = token_inverted_map[highest_score_idx]
            if highest_score_token not in blocks:
                blocks[highest_score_token] = Block(highest_score_token)
            blocks[highest_score_token].append(ref_id, self._all_data[ref_id].source)
        self._blocks.extend(blocks.values())
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

    def update_candidate_pairs(self, generate_deduplication_pairs: bool = True) -> None:
        self._candidates.clear()
        for block in self._blocks:
            for ref_id1, ref_id2 in block.candidate_pairs(generate_deduplication_pairs):
                self._candidates.append(
                    (self._all_data[ref_id1], self._all_data[ref_id2])
                )

    def candidate_pairs(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        yield from (
            tuple(map(lambda x: x.reference, candidate)) for candidate in self._candidates
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
