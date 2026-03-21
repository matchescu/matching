"""Dataset and data loading utilities for entity matching"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from matchescu.data import Record
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier


class DeepMatcherDataset(Dataset):
    """Dataset for entity matching with multiple attributes"""

    __DEFAULT_TOKENIZER_NAME = "google-bert/bert-base-uncased"

    def __init__(
        self,
        id_table: IdTable,
        split: Split,
        attr_map: dict[str, str] = None,
        exclude_from_comparison: Iterable[str | int] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        max_len: int = 30,
    ):
        self.__id_table = id_table
        self.__pairs, self.__labels = split.to_comparison_labels(self.__id_table)
        self.__max_len = max_len
        self.__attr_map = attr_map
        self.__tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            DeepMatcherDataset.__DEFAULT_TOKENIZER_NAME,
        )
        self.__max_len = max_len
        left, right = next(iter(self.__pairs))
        min_attr_count = min(len(left), len(right))
        attr_map = attr_map or {i: i for i in range(min_attr_count)}
        exclude = set(exclude_from_comparison or ())
        self.__attr_map = {
            l_attr: r_attr
            for l_attr, r_attr in attr_map.items()
            if left not in exclude and right not in exclude
        }
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
        left_tokens = []
        right_tokens = []

        for left_attr, right_attr in self.__attr_map.items():
            left_text = str(left[left_attr])
            left_enc = self.__tokenizer(
                left_text,
                padding="max_length",
                truncation=True,
                max_length=self.__max_len,
                return_tensors=None,
            )
            right_text = str(right[right_attr])
            right_enc = self.__tokenizer(
                right_text,
                padding="max_length",
                truncation=True,
                max_length=self.__max_len,
                return_tensors=None,
            )
            left_tokens.append(left_enc["input_ids"])
            right_tokens.append(right_enc["input_ids"])
        left_attrs = torch.tensor(left_tokens, dtype=torch.long)
        right_attrs = torch.tensor(right_tokens, dtype=torch.long)
        return {
            "left_attrs": left_attrs,
            "right_attrs": right_attrs,
            "label": torch.tensor(self.__labels[idx], dtype=torch.float),
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
