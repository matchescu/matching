from matchescu.extraction import Traits
from pathlib import Path

from matchescu.matching.evaluation.data.benchmark._base import BenchmarkDataFactory
from matchescu.matching.evaluation.data.benchmark._csv import CsvBenchmarkData
from matchescu.matching.evaluation.data.benchmark._magellan import (
    MagellanBenchmarkData,
    MagellanTraits,
)

from matchescu.matching.config import (
    TraitConfig,
    MagellanBenchmarkDataConfig,
    CsvBenchmarkDataConfig,
)


def get_traits(configs: list[TraitConfig]) -> Traits:
    """Initialize ``Traits`` from ``TraitConfig`` objects.

    :param configs: configs to use

    :return: ``Traits`` instance fully initialized from configs.Ï
    """
    traits = Traits()
    for cfg in configs:
        method = getattr(traits, cfg.type, None)
        if method is None:
            raise ValueError(f"Unknown trait type: '{cfg.type}'")
        traits = method(cfg.keys)
    return traits


class CsvBenchmarkDataFactory(BenchmarkDataFactory[CsvBenchmarkData]):
    """Create fully loaded ``CsvBenchmarkData`` from config params."""

    def __init__(self, params: CsvBenchmarkDataConfig) -> None:
        self._params = params

    def create(self, root_data_dir: Path | None = None) -> CsvBenchmarkData:
        data_dir = Path(self._params.directory)
        if not data_dir.is_absolute() and root_data_dir is not None:
            data_dir = root_data_dir / data_dir
        data = CsvBenchmarkData(data_dir, [fp.file_name for fp in self._params.sources])
        traits_map = {
            fp.file_name: get_traits(fp.traits) for fp in self._params.sources
        }

        id_cols = {fp.file_name: fp.id_col for fp in self._params.sources}
        source_cols = {fp.file_name: fp.source_col for fp in self._params.sources}
        has_headers = {fp.file_name: fp.has_header for fp in self._params.sources}

        data.load_data(
            traits=traits_map,
            id_cols=id_cols,
            source_cols=source_cols,
            headers=has_headers,
        )
        data = self._load_ideal_mapping(data)
        data = self._load_clusters(data)
        data = self._perform_split(data)

        return data

    def _load_existing_splits(self, data: CsvBenchmarkData) -> CsvBenchmarkData:
        # todo load splits from files
        return data

    def _perform_split(self, data: CsvBenchmarkData) -> CsvBenchmarkData:
        if (split_config := self._params.splits) is not None:
            if self._params.splits.files is not None:
                return self._load_existing_splits(data)
            else:
                return data.split(
                    split_config.ratios,
                    split_config.neg_pos_ratio,
                    split_config.match_bridge_ratio,
                    split_config.max_total_sample_count,
                    save=True,
                )
        else:
            return data.split(save=True)

    def _load_ideal_mapping(self, data: CsvBenchmarkData) -> CsvBenchmarkData:
        pwm = self._params.pairwise_mapping
        if isinstance(pwm, str):
            return data.with_ideal_mapping(
                mapping_file=pwm,
                id_cols=(0, 1),
                source_cols=None,
                label_col=2,
                has_header=True,
            )
        else:
            source_cols = None
            if pwm.left_source_col is not None and pwm.right_source_col is not None:
                source_cols = (pwm.left_source_col, pwm.right_source_col)
            label_col = -1
            if pwm.label_col is not None:
                label_col = pwm.label_col
            return data.with_ideal_mapping(
                mapping_file=pwm.file_name,
                id_cols=(pwm.left_id_col, pwm.right_id_col),
                source_cols=source_cols,
                label_col=label_col,
                has_header=pwm.has_header,
            )

    def _load_clusters(self, data: CsvBenchmarkData) -> CsvBenchmarkData:
        if isinstance(self._params.cluster_mapping, str):
            return data.with_clusters(self._params.cluster_mapping, has_headers=True)
        else:
            return data.with_clusters(
                self._params.cluster_mapping.file_name,
                self._params.cluster_mapping.id_col,
                self._params.cluster_mapping.source_col,
                self._params.cluster_mapping.label_col,
                self._params.cluster_mapping.has_header,
            )


class MagellanBenchmarkDataFactory(BenchmarkDataFactory[MagellanBenchmarkData]):
    """Initialize fully loaded ``MagellanBenchmarkData`` from config."""

    _MAGELLAN_TRAITS = MagellanTraits()

    def __init__(self, params: MagellanBenchmarkDataConfig) -> None:
        self._params = params

    def create(self, root_data_dir: Path | None = None) -> MagellanBenchmarkData:
        data_dir = Path(self._params.directory)
        if not data_dir.is_absolute() and root_data_dir is not None:
            data_dir = root_data_dir / data_dir
        data = MagellanBenchmarkData(data_dir)
        left_traits = self._resolve_traits(self._params.left_traits)
        right_traits = (
            self._resolve_traits(self._params.right_traits)
            if self._params.right_traits is not None
            else left_traits
        )

        data.load_left(left_traits).load_right(right_traits).load_splits()
        return data

    @classmethod
    def _resolve_traits(cls, trait_configs: str | list[TraitConfig]):
        """A string is looked up in the well-known MagellanTraits dict;
        a list of TraitConfig is built manually."""
        if isinstance(trait_configs, str):
            return cls._MAGELLAN_TRAITS[trait_configs]
        return get_traits(trait_configs)
