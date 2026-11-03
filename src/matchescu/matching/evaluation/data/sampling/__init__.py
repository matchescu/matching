"""Deterministic extraction and sampling of declared bad triplets.

The package exists so that dataset preparation and comparison space
generation can share one proven implementation of bad-triplet sampling.
"""

from ._bad_triplets import (
    BadTriplet,
    BadTripletSample,
    BadTripletSampler,
    BridgeIndex,
    ExclusionReport,
)

__all__ = [
    "BadTriplet",
    "BadTripletSample",
    "BadTripletSampler",
    "BridgeIndex",
    "ExclusionReport",
]
