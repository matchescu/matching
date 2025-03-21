from dataclasses import dataclass

from matchescu.reference_store.comparison_space._in_memory import (
    InMemoryComparisonSpace,
)
from matchescu.typing import EntityReferenceIdentifier


@dataclass
class BlockingMetrics:
    pair_completeness: float
    pair_quality: float
    reduction_ratio: float


def calculate_metrics(
    comparison_space: InMemoryComparisonSpace,
    ground_truth: set[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]],
    initial_size: int,
) -> BlockingMetrics:
    candidate_ids = set(comparison_space)
    true_positive_pairs = ground_truth.intersection(candidate_ids)
    pc = len(true_positive_pairs) / len(ground_truth) if ground_truth else 0
    pq = len(true_positive_pairs) / len(candidate_ids) if candidate_ids else 0
    candidate_ratio = len(candidate_ids) / initial_size if initial_size != 0 else 0
    return BlockingMetrics(pc, pq, 1 - candidate_ratio)
