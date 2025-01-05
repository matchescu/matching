from matchescu.matching.ml.datasets._deduplication import DeduplicationDataSet
from matchescu.matching.ml.datasets._record_linkage import RecordLinkageDataSet
from matchescu.matching.ml.datasets._reference_comparison import (
    AttributeComparison,
    PatternEncodedComparison,
)
from matchescu.matching.ml.datasets._torch import PlTorchDataset


__all__ = [
    "DeduplicationDataSet",
    "RecordLinkageDataSet",
    "AttributeComparison",
    "PatternEncodedComparison",
    "PlTorchDataset",
]
