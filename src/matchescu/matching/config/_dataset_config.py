from os import PathLike
from typing import Literal, Annotated, Union

from pydantic import Field

from matchescu.matching.config import ConfigModel


class TraitConfig(ConfigModel):
    type: str = Field(default="string")
    keys: list[int | str] = Field(default_factory=list)


class BenchmarkDataConfig(ConfigModel):
    """Base class for configuring datasets."""

    directory: str | PathLike


class MagellanBenchmarkDataConfig(BenchmarkDataConfig):
    type: Literal["magellan"] = "magellan"
    left_traits: str | list[TraitConfig]
    right_traits: str | list[TraitConfig] | None = None


class CsvFileConfig(ConfigModel):
    file_name: str
    has_header: bool = True


class SourceFileConfig(CsvFileConfig):
    traits: list[TraitConfig] = Field(default_factory=list)
    id_col: str | int = 0
    source_col: str | int | None = None


class PairwiseGroundTruthConfig(CsvFileConfig):
    left_id_col: str | int = 0
    right_id_col: str | int = 1
    left_source_col: str | int | None = None
    right_source_col: str | int | None = None
    label_col: str | int | None = None


class ClusterGroundTruthConfig(CsvFileConfig):
    id_col: str | int = 0
    source_col: str | int = 1
    label_col: str | int = 2


class SplitConfig(ConfigModel):
    files: dict[str, str | PairwiseGroundTruthConfig] | None = None
    ratios: dict[str, float] | None = None
    max_total_sample_count: int | None = None
    neg_pos_ratio: float = 8.0
    match_bridge_ratio: float = 2.0


class CsvBenchmarkDataConfig(BenchmarkDataConfig):
    type: Literal["csv"] = "csv"
    sources: list[SourceFileConfig]
    pairwise_mapping: str | PairwiseGroundTruthConfig
    cluster_mapping: str | ClusterGroundTruthConfig | None = None
    splits: SplitConfig | None = None


AnyDatasetConfig = Annotated[
    Union[MagellanBenchmarkDataConfig, CsvBenchmarkDataConfig],
    Field(discriminator="type"),
]
