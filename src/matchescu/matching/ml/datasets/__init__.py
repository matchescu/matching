from matchescu.matching.ml.datasets._deduplication import DeduplicationDataSet
from matchescu.matching.ml.datasets._record_linkage import RecordLinkageDataSet
from matchescu.matching.ml.datasets._sampling import (
    AttributeComparison,
    PatternEncodedComparison,
)


__all__ = [
    "DeduplicationDataSet",
    "RecordLinkageDataSet",
    "AttributeComparison",
    "PatternEncodedComparison",
]
