from abc import ABC
from pathlib import Path
from typing import Iterable


class BenchmarkData(ABC):
    @staticmethod
    def _check_files(paths: list[str | Path]) -> Iterable[Path]:
        for path in map(Path, paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            yield path.absolute()
