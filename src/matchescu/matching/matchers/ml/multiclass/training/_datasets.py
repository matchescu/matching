import random

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding

from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.reference_store.id_table import IdTable

from .._encoder import to_ditto_text
from ...training import MatchescuDataset


class AsymmetricMultiClassDataset(MatchescuDataset):
    _LABEL_SWAP = {0: 0, 1: 1, 2: 3, 3: 2}

    def __init__(
        self,
        id_table: IdTable,
        split: Split,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 256,
        left_cols: tuple | None = None,
        right_cols: tuple | None = None,
        random_seed: int = 42,
    ):
        super().__init__(id_table, split)
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__left_cols = left_cols
        self.__right_cols = right_cols
        self.__label_counts = np.bincount(self._labels)
        random.seed(random_seed)

    @property
    def label_counts(self) -> np.ndarray:
        return self.__label_counts

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            encodings, attention masks and token type ids for both normal
            and reversed input orders.
        """
        left, right = self._pairs[idx]
        left_text = to_ditto_text(left)
        right_text = to_ditto_text(right)

        y = self._labels[idx]

        x_fwd = {
            k: v.squeeze(0)
            for k, v in self.__tokenizer(
                text=left_text,
                text_pair=right_text,
                max_length=self.__max_len,
                truncation=True,
                return_tensors="pt",
            ).items()
        }
        x_rev = {
            k: v.squeeze(0)
            for k, v in self.__tokenizer(
                text=right_text,
                text_pair=left_text,
                max_length=self.__max_len,
                truncation=True,
                return_tensors="pt",
            ).items()
        }

        return x_fwd, x_rev, y

    def _pad(self, batch: BatchEncoding) -> dict[str, torch.LongTensor]:
        return {
            k: v
            for k, v in self.__tokenizer.pad(
                batch,
                max_length=self.__max_len,
                padding=True,
                return_tensors="pt",
            ).items()
        }

    def _collate(self, batch: list[tuple]) -> tuple[dict, dict, torch.LongTensor]:
        x_fwd_list, x_rev_list, y = zip(*batch)
        x_fwd_padded = self._pad(x_fwd_list)
        x_rev_padded = self._pad(x_rev_list)
        return x_fwd_padded, x_rev_padded, torch.LongTensor(y)
