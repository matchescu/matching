import itertools
import random
from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerFast

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
        augmentation_probability: float = 0.5,
        left_cols: tuple | None = None,
        right_cols: tuple | None = None,
        random_seed: int = 42,
    ):
        super().__init__(id_table, split)
        self.__tokenizer = tokenizer
        self.__aug_prob = augmentation_probability
        self.__max_len = max_len
        self.__left_cols = left_cols
        self.__right_cols = right_cols
        random.seed(random_seed)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left, right = self._pairs[idx]
        y = self._labels[idx]
        if random.random() < self.__aug_prob:
            left, right = right, left
            y = self._LABEL_SWAP[y]

        left_text = to_ditto_text(left, self.__left_cols)
        right_text = to_ditto_text(right, self.__right_cols)
        x = self.__tokenizer.encode(
            text=left_text,
            text_pair=right_text,
            max_length=self.__max_len,
            truncation=True,
        )
        return x, y

    @staticmethod
    def __pad(x: Sequence, total_length: int) -> torch.LongTensor:
        tensor_data = list(
            map(
                lambda vec: list(
                    itertools.chain(
                        vec, itertools.repeat(0, max(total_length - len(vec), 0))
                    )
                ),
                x,
            )
        )
        return torch.LongTensor(tensor_data)

    def _collate(self, batch: list[tuple]) -> tuple[torch.LongTensor, ...]:
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            n = max(map(len, x1 + x2))
            x1 = AsymmetricMultiClassDataset.__pad(x1, n)
            x2 = AsymmetricMultiClassDataset.__pad(x2, n)
            return x1, x2, torch.LongTensor(y)
        else:
            x, y = zip(*batch)
            n = max(map(len, x))
            x = AsymmetricMultiClassDataset.__pad(x, n)
            return x, torch.LongTensor(y)
