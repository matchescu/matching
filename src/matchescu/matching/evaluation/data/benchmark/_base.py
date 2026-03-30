from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, TypeVar, Generic

from matchescu.reference_store.id_table import InMemoryIdTable, IdTable


class BenchmarkData(ABC):
    _SPLIT_NAMES = ["train", "valid", "test"]

    def __init__(self):
        self._splits = {}
        self._id_table = InMemoryIdTable()

    @property
    def id_table(self) -> IdTable:
        return self._id_table

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

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
