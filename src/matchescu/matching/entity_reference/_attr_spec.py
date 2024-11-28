from dataclasses import dataclass

from matchescu.matching.attribute import AttrMatchCallable


@dataclass
class AttrComparisonSpec:
    label: str
    left_ref_key: int | str
    right_ref_key: int | str
    match_strategy: AttrMatchCallable
