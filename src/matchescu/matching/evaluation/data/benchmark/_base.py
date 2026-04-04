from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, TypeVar, Generic

from matchescu.reference_store.id_table import InMemoryIdTable, IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId


class BenchmarkData(ABC):
    _SPLIT_NAMES = ["train", "valid", "test"]

    def __init__(self):
        self._splits = {}
        self._id_table: IdTable = InMemoryIdTable()
        self._match_gt: dict[tuple[RefId, RefId], int] = {}
        self._cluster_gt: dict[RefId, int] = {}

    @property
    def id_table(self) -> IdTable:
        return self._id_table

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def true_matches(self) -> dict[tuple[RefId, RefId], int]:
        return self._match_gt

    @property
    def ref_id_cluster_map(self) -> dict[RefId, int]:
        return self._cluster_gt

    def compute_clusters(self) -> frozenset[frozenset[RefId]]:
        clusters = {}
        for ref_id, cluster_no in self._cluster_gt.items():
            clusters.setdefault(cluster_no, set()).add(ref_id)
        return frozenset(
            frozenset(ref_id for ref_id in cluster)
            for cluster_no, cluster in clusters.items()
        )

    @staticmethod
    def _check_files(paths: list[str | Path]) -> Iterable[Path]:
        for path in map(Path, paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            yield path.absolute()

    def __getattr__(self, item: str):
        if item in self._splits:
            return self._splits[item]
        raise AttributeError(f"{item} split not found")

    def __dir__(self):
        return sorted([*super().__dir__(), *self._splits.keys()])


T = TypeVar("T", bound=BenchmarkData)


class BenchmarkDataFactory(Generic[T], ABC):
    @abstractmethod
    def create(self, root_data_dir: Path | None = None) -> T:
        raise NotImplementedError
