from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RecordLinkageConfig:
    left_id: str
    right_id: str
    ground_truth_label_col: str
    col_comparison_config: Sequence[tuple[str, str]] = field(default_factory=list)
