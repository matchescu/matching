from pathlib import Path
from typing import Iterable

from matchescu.data_sources import CsvDataSource
from matchescu.extraction import RecordExtraction, single_record, Traits
from matchescu.typing import Record, EntityReferenceIdentifier as RefId


class CsvRecordExtraction(RecordExtraction):
    def __init__(
        self,
        fpath: str | Path,
        traits: Traits,
        id_col: str | int | None = None,
        source_attr: str | int | None = None,
        has_header: bool = True,
    ):
        self.__fpath = Path(fpath).absolute()
        self.__source_attr = source_attr
        self.__id_attr = id_col
        self.__ds = CsvDataSource(self.__fpath, list(traits), has_header=has_header)
        self.__ds.read()
        super().__init__(self.__ds, self._id_factory, single_record)

    def _id_factory(self, records: Iterable[Record]) -> RefId:
        record = next(iter(records))
        try:
            label = record[self.__id_attr]
        except ValueError:
            label = record[0]
        try:
            source = record[self.__source_attr]
        except ValueError:
            source = self.__ds.name

        return RefId(label=label, source=source)

    @property
    def data_source(self) -> CsvDataSource:
        return self.__ds
