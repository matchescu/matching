import itertools
from os import PathLike
from pathlib import Path

import networkx as nx
import polars as pl
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import Traits, RecordExtraction, single_record
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.reference_store.id_table import InMemoryIdTable, IdTable
from matchescu.typing import (
    EntityReferenceIdFactory,
    EntityReferenceIdentifier as RefId,
)


class MagellanDataset:
    def __init__(self, folder_path: str | PathLike) -> None:
        self.__dataset_dir = Path(folder_path)
        if not self.__dataset_dir.is_dir():
            raise ValueError(f"'{self.__dataset_dir}' is not a directory")
        self.__left_table_path = self.__dataset_dir / "tableA.csv"
        self.__right_table_path = self.__dataset_dir / "tableB.csv"
        self.__train_path = self.__dataset_dir / "train.csv"
        self.__valid_path = self.__dataset_dir / "valid.csv"
        self.__test_path = self.__dataset_dir / "test.csv"
        for path in (
            self.__left_table_path,
            self.__right_table_path,
            self.__train_path,
            self.__valid_path,
            self.__test_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.__id_table: IdTable = InMemoryIdTable()
        self.__left_source = self.__left_table_path.stem
        self.__right_source = self.__right_table_path.stem
        self._train: Split | None = None
        self._valid: Split | None = None
        self._test: Split | None = None

    def _load_csv_table(
        self, path: Path, traits: Traits, id_factory: EntityReferenceIdFactory
    ) -> str:
        ds = CsvDataSource(path, list(traits)).read()
        re = RecordExtraction(ds, id_factory, single_record)
        for ref in list(re()):
            self.__id_table.put(ref)
        return ds.name

    def load_left(
        self, traits: Traits, id_factory: EntityReferenceIdFactory
    ) -> "MagellanDataset":
        self.__left_source = self._load_csv_table(
            self.__left_table_path, traits, id_factory
        )
        return self

    def load_right(
        self, traits: Traits, id_factory: EntityReferenceIdFactory
    ) -> "MagellanDataset":
        self.__right_source = self._load_csv_table(
            self.__right_table_path, traits, id_factory
        )
        return self

    def __load_split(self, path: Path) -> Split:
        rows = list(
            itertools.starmap(
                lambda left, right, is_match: (
                    RefId(left, self.__left_source),
                    RefId(right, self.__right_source),
                    is_match,
                ),
                pl.read_csv(path, ignore_errors=True).iter_rows(named=False),
            )
        )
        gt = set((left, right) for left, right, is_match in rows if is_match == 1)
        cs = InMemoryComparisonSpace()
        for left, right, _ in rows:
            cs.put(left, right)
        clusters = {
            ix: comp for ix, comp in enumerate(nx.connected_components(nx.Graph(gt)), 1)
        }
        return Split(cs, {cmp: 1 for cmp in gt}, clusters)

    def load_splits(self) -> "MagellanDataset":
        if not self.__left_source or not self.__right_source:
            raise ValueError(
                "left + right data sources must be loaded before loading splits"
            )
        self._train = self.__load_split(self.__train_path)
        self._valid = self.__load_split(self.__valid_path)
        self._test = self.__load_split(self.__test_path)
        return self

    @property
    def id_table(self) -> IdTable:
        return self.__id_table

    @property
    def left_source(self) -> str:
        return self.__left_source

    @property
    def right_source(self) -> str:
        return self.__right_source

    @property
    def train_split(self) -> Split:
        return self._train

    @property
    def valid_split(self) -> Split:
        return self._valid

    @property
    def test_split(self) -> Split:
        return self._test

    def all_data(self) -> Split:
        return Split.merge([self._train, self._test, self._valid])


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

    def __getattr__(self, item: str) -> Traits:
        return self[item]

    def __getitem__(self, item: str) -> Traits:
        assert isinstance(item, str)
        key = item.upper()
        assert key in self.__TRAIT_DICT
        return self.__TRAIT_DICT[key]
