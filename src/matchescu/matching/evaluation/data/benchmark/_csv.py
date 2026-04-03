from pathlib import Path
from typing import Iterable, Mapping, Union

from matchescu.extraction import Traits
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.matching.config import CsvBenchmarkDataConfig
from matchescu.matching.evaluation.data.extraction import CsvRecordExtraction
from matchescu.matching.evaluation.data.splits import SplitGenerator
from matchescu.matching.evaluation.ground_truth import (
    read_pairwise_mapping_csv,
    read_clusters_csv,
    EquivalenceClassPartitioner,
)

from ._base import BenchmarkData, BenchmarkDataFactory
from ._config_adapters import get_traits


class CsvBenchmarkData(BenchmarkData):
    __DEFAULT_SPLIT_RATIOS = {"train": 3, "valid": 1, "test": 1}

    def __init__(self, data_dir: Path, source_files: list[str]):
        super().__init__()
        self._data_dir = data_dir
        paths = [data_dir / src for src in source_files]
        self._paths = list(self._check_files(paths))

    @property
    def name(self) -> str:
        return self._data_dir.stem

    def load_data(
        self,
        traits: Iterable[Traits] | Mapping[str, Traits],
        id_cols: Iterable[str | int] | Mapping[str, str | int] | None = None,
        source_cols: Iterable[str | int] | Mapping[str, str | int] | None = None,
        headers: Iterable[bool] | Mapping[str, bool] | None = None,
    ) -> "CsvBenchmarkData":
        if not traits:
            raise ValueError("must provide traits")
        file_traits = self._get_file_params(traits, Traits)
        file_id_cols = {p: None for p in self._paths}
        if id_cols is not None:
            file_id_cols = self._get_file_params(id_cols, Union[str, int])
        file_source_cols = {p: None for p in self._paths}
        if source_cols is not None:
            file_source_cols = self._get_file_params(source_cols, Union[str, int])
        file_headers = {p: True for p in self._paths}
        if headers is not None:
            file_headers = self._get_file_params(headers, bool)
        self._build_id_table(file_traits, file_id_cols, file_source_cols, file_headers)
        return self

    def with_ideal_mapping(
        self,
        mapping_file: str,
        id_cols: tuple[str | int, str | int] = None,
        source_cols: tuple[str | int, str | int] = None,
        label_col: str | int | None = None,
        has_header: bool = False,
    ) -> "CsvBenchmarkData":
        mapping_path = self._data_dir / mapping_file
        sources = [] if source_cols is not None else [p.stem for p in self._paths]
        self._match_gt = read_pairwise_mapping_csv(
            mapping_path,
            *sources,
            id_cols=id_cols,
            source_cols=source_cols,
            label_col=label_col,
            has_header=has_header,
        )
        return self

    def with_clusters(
        self,
        clusters_file: str | None = None,
        id_col: str | int = 0,
        source_col: str | int = 1,
        cluster_label_col: str | int = 2,
        has_headers: bool = True,
    ) -> "CsvBenchmarkData":
        if clusters_file is not None:
            self._cluster_gt = {
                ref_id: cluster_no
                for cluster_no, cluster in read_clusters_csv(
                    self._data_dir / clusters_file,
                    has_header=has_headers,
                    id_col=id_col,
                    source_col=source_col,
                    label_col=cluster_label_col,
                    source_name=self._paths[0].stem,
                ).items()
                for ref_id in cluster
            }
        else:
            if len(set(self._match_gt.values())) > 2:
                raise ValueError(
                    "missing cluster labels for multi-class pairwise mappings"
                )
            ecp = EquivalenceClassPartitioner(self._id_table.ids())
            try:
                self._cluster_gt = {
                    ref_id: idx
                    for idx, cluster in enumerate(ecp(self._match_gt), 1)
                    for ref_id in cluster
                }
            except KeyError as e:
                raise KeyError("could not find item in ID table") from e
        return self

    def split(
        self,
        split_ratios: Mapping[str, int] | None = None,
        neg_pos_ratio: float = 8.0,
        bridge_class_ratio: float = 4.0,
        max_sample_count: int | None = None,
        save: bool = False,
    ) -> "CsvBenchmarkData":
        if self._id_table is None:
            raise RuntimeError("call load_data() before calling split()")
        if self._match_gt is None:
            raise RuntimeError("call with_ideal_mapping() before calling split()")
        if self._cluster_gt is None:
            raise RuntimeError("call with_clusters() before calling split()")
        splitter = (
            SplitGenerator(
                split_ratios or self.__DEFAULT_SPLIT_RATIOS,
                neg_pos_ratio,
                bridge_class_ratio,
                max_sample_count,
            )
            .load(self._id_table, self._cluster_gt, self._match_gt)
            .generate()
        )
        if save:
            splitter.save(self._data_dir)
        self._splits = {f"{k}_split": v for k, v in splitter.get_splits().items()}
        return self

    @property
    def comparison_space_size(self) -> int:
        n = len(self._id_table)
        return (n * (n - 1)) // 2

    @property
    def ideal_mapping_size(self) -> int:
        return len(self._match_gt)

    @property
    def cluster_count(self) -> int:
        return len(self._cluster_gt)

    def _build_id_table(
        self,
        file_traits: dict[Path, Traits],
        file_id_cols: dict[Path, str | int | None],
        file_source_cols: dict[Path, str | int | None],
        file_has_headers: dict[Path, bool],
    ):
        self._id_table = InMemoryIdTable()
        for path, traits in file_traits.items():
            id_col = file_id_cols[path]
            source_col = file_source_cols[path]
            has_headers = file_has_headers[path]
            extract = CsvRecordExtraction(path, traits, id_col, source_col, has_headers)
            for ref in extract():
                self._id_table.put(ref)

    def _get_file_params[T](
        self, params: Iterable[T] | Mapping[str, T], item_type: type[T]
    ) -> dict[Path, T]:
        if isinstance(params, Mapping):
            return self._get_file_param_from_mapping(params)
        elif isinstance(params, Iterable):
            return self._get_file_param_from_iterable(params, item_type)
        else:
            raise ValueError(f"unsupported traits input type: {type(params)}")

    def _get_file_param_from_iterable[T](
        self, param: Iterable[T], item_type: type[T]
    ) -> dict[Path, T]:
        lst = list(param)
        has_incorrect_type = any(not isinstance(x, item_type) for x in lst)
        if has_incorrect_type:
            raise ValueError(f"not all items in sequence were {item_type}")
        if (len(lst) != 1) ^ (len(lst) != len(self._paths)):
            raise ValueError(f"sequence must be of length 1 or {len(self._paths)}")
        if len(lst) == 1:
            return {p: lst[0] for p in self._paths}
        else:
            return dict(zip(self._paths, lst))

    def _get_file_param_from_mapping[T](self, param: Mapping[str, T]) -> dict[Path, T]:
        missing_keys = [x.name for x in self._paths if x.name not in param]
        if len(param) != len(self._paths) or len(missing_keys) > 0:
            raise ValueError(
                f"mapping is missing values for: {", ".join(missing_keys)}"
            )
        return {p: param[p.name] for p in self._paths}


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
