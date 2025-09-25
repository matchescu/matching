from dataclasses import dataclass
from typing import Generic

from matchescu.matching.similarity import T, Similarity


@dataclass
class AttrCmpConfig(Generic[T]):
    left: str | int
    right: str | int
    agreement_levels: list[T]
    sim: Similarity[T]


@dataclass(frozen=True)
class RecordLinkageConfig:
    left_id: str
    right_id: str
    ground_truth_label_col: str
    col_comparison_config: list[AttrCmpConfig[float]]
