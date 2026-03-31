from dataclasses import dataclass


@dataclass(frozen=True)
class MatchResult:
    label: int
    confidence: float
