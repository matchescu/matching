from enum import IntEnum


class TernaryResult(IntEnum):
    """Enumerates possible match results inspired by the Fellegi-Sunter model."""

    NonMatch = 0
    Match = 1
    NoComparisonData = 2


class BinaryResult(IntEnum):
    """Enumerates values that can be used with a Naive Bayes classifier."""

    Negative = -1
    Positive = 1
