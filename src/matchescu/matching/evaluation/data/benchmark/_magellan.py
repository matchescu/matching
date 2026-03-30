import itertools
from os import PathLike
from pathlib import Path

import polars as pl
from matchescu.extraction import Traits
from matchescu.matching.evaluation.data.benchmark._base import BenchmarkData
from matchescu.matching.evaluation.data.extraction._record_extraction import (
    CsvRecordExtraction,
)
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.matching.evaluation.ground_truth import EquivalenceClassPartitioner
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier as RefId


class MagellanBenchmarkData(BenchmarkData):
    def __init__(self, folder_path: str | PathLike) -> None:
        super().__init__()
        self.__dataset_dir = Path(folder_path)
        if not self.__dataset_dir.is_dir():
            raise ValueError(f"'{self.__dataset_dir}' is not a directory")
        self.__left_table_path = self.__dataset_dir / "tableA.csv"
        self.__left_source = self.__left_table_path.stem
        self.__right_table_path = self.__dataset_dir / "tableB.csv"
        self.__right_source = self.__right_table_path.stem
        self.__split_paths = [
            self.__dataset_dir / f"{split}.csv" for split in self._SPLIT_NAMES
        ]
        self._check_files(
            [
                self.__left_table_path,
                self.__right_table_path,
                *self.__split_paths,
            ]
        )

    def _load_csv_table(self, path: Path, traits: Traits) -> str:
        extract = CsvRecordExtraction(path, traits)
        for ref in extract():
            self._id_table.put(ref)
        return extract.data_source.name

    def load_left(self, traits: Traits) -> "MagellanBenchmarkData":
        self.__left_source = self._load_csv_table(self.__left_table_path, traits)
        return self

    def load_right(self, traits: Traits) -> "MagellanBenchmarkData":
        self.__right_source = self._load_csv_table(self.__right_table_path, traits)
        return self

    def __load_split(self, path: Path) -> Split:
        rows = list(
            itertools.starmap(
                lambda x, y, is_match: (
                    RefId(x, self.__left_source),
                    RefId(y, self.__right_source),
                    is_match,
                ),
                pl.read_csv(path, ignore_errors=True).iter_rows(named=False),
            )
        )
        cs = InMemoryComparisonSpace()
        ids = {}
        for left, right, _ in rows:
            cs.put(left, right)
            ids[left] = None
            ids[right] = None

        matcher_gt = {(left, right): label for left, right, label in rows if label == 1}
        ecp = EquivalenceClassPartitioner(ids)
        clusters = {
            cluster_no: set(cluster)
            for cluster_no, cluster in enumerate(ecp(matcher_gt), 1)
        }
        return Split(cs, matcher_gt, clusters)

    def load_splits(self) -> "MagellanBenchmarkData":
        if not self.__left_source or not self.__right_source:
            raise ValueError(
                "left + right data sources must be loaded before loading splits"
            )
        self._splits = {
            f"{name}_split": self.__load_split(path)
            for name, path in zip(self._SPLIT_NAMES, self.__split_paths)
        }
        return self

    @property
    def name(self) -> str:
        return self.__dataset_dir.stem

    @property
    def left_source(self) -> str:
        return self.__left_source

    @property
    def right_source(self) -> str:
        return self.__right_source

    def __getattr__(self, item: str):
        if item in self._splits:
            return self._splits[item]
        raise AttributeError(f"{item} split not found")

    def __dir__(self):
        return sorted([*super().__dir__(), *self._splits.keys()])

    def all_data(self) -> Split:
        return Split.merge(list(self._splits.values()))


class MagellanTraits:
    __TRAIT_DICT = {
        "ABT-BUY": Traits().string(["name", "description"]).currency(["price"]),
        "AMAZON-GOOGLE": Traits().string(["title", "manufacturer"]).currency(["price"]),
        "BEER": Traits().string(["Beer_Name", "Brew_Factory_Name", "Style"]),
        "DBLP-ACM": Traits().string(["title", "authors", "venue"]).int(["year"]),
        "DBLP-SCHOLAR": Traits().string(["title", "authors", "venue", "year"]),
        "FODORS-ZAGAT": Traits()
        .string(["name", "addr", "city", "phone", "type"])
        .int(["class"]),
        "ITUNES-AMAZON": Traits().string(
            [
                "Song_Name",
                "Artist_Name",
                "Album_Name",
                "Genre",
                "Price",
                "CopyRight",
                "Time",
                "Released",
            ]
        ),
        "WALMART-AMAZON": Traits()
        .string(["title", "category", "brand", "modelno"])
        .currency(["price"]),
    }

    def __getitem__(self, item: str) -> Traits:
        assert isinstance(item, str)
        key = item.upper()
        assert key in self.__TRAIT_DICT
        return self.__TRAIT_DICT[key]
