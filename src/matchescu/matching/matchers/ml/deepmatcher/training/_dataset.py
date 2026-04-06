"""Dataset and data loading utilities for entity matching"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from matchescu.data import Record
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.matching.matchers.ml.deepmatcher._encoder import (
    to_deepmatcher_repr,
    ensure_attr_map,
)
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier


class DeepMatcherDataset(Dataset):
    """Dataset for entity matching with multiple attributes"""

    __DEFAULT_TOKENIZER_NAME = "google-bert/bert-base-uncased"

    def __init__(
        self,
        id_table: IdTable,
        split: Split,
        tokenizer: PreTrainedTokenizerBase = None,
        attr_map: dict[str, str] = None,
        exclude_from_comparison: Iterable[str | int] = None,
        max_len: int = 30,
    ):
        self.__id_table = id_table
        self.__pairs, self.__labels = split.to_comparison_labels(self.__id_table)
        self.__tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            DeepMatcherDataset.__DEFAULT_TOKENIZER_NAME,
        )
        self.__max_len = max_len
        left, right = next(iter(self.__pairs))
        self.__attr_map = ensure_attr_map(
            left, right, attr_map, exclude_from_comparison
        )
        self.__attr_count = len(self.__attr_map)

    @property
    def attr_count(self) -> int:
        return self.__attr_count

    @property
    def attr_dims(self) -> int:
        return self.__max_len

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        left, right = self.__pairs[idx]
        return {
            "label": torch.tensor(self.__labels[idx], dtype=torch.float),
            **to_deepmatcher_repr(
                left, right, self.__tokenizer, self.__attr_map, self.__max_len
            ),
        }

    @staticmethod
    def __dl_collate_fn(dataset_record):
        return {
            "left_attrs": torch.stack([item["left_attrs"] for item in dataset_record]),
            "right_attrs": torch.stack(
                [item["right_attrs"] for item in dataset_record]
            ),
            "label": torch.tensor([item["label"] for item in dataset_record]),
        }

    def get_data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__dl_collate_fn,
        )


if __name__ == "__main__":
    from matchescu.matching.evaluation.data import MagellanBenchmarkData, MagellanTraits

    traits = MagellanTraits().beer

    ds = MagellanBenchmarkData("data/magellan/beer")

    def _id_factory(rows: Iterable[Record], source: str) -> EntityReferenceIdentifier:
        return EntityReferenceIdentifier(next(iter(rows))["id"], source)

    ds.load_left(traits)
    ds.load_right(traits)
    ds.load_splits()

    dmds = DeepMatcherDataset(ds.id_table, ds.train_split)
    for item in dmds.get_data_loader():
        print(item)
