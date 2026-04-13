from abc import abstractmethod
from typing import Sized, TypeVar

import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

from matchescu.matching.evaluation.data.splits import Split
from matchescu.reference_store.id_table import IdTable


class MatchescuDataset(Dataset, Sized):
    def __init__(self, id_table: IdTable, split: Split):
        self._id_table = id_table
        self._pairs, self._labels = split.to_comparison_labels(self._id_table)

    def __len__(self):
        return len(self._labels)

    @abstractmethod
    def _collate(self, record):
        raise NotImplementedError

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        class_counts = np.bincount(self._labels)
        sample_weights = 1.0 / class_counts[self._labels]
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(self), replacement=True
        )

    def get_data_loader(
        self, batch_size: int = 32, shuffle: bool = True, sampler: Sampler | None = None
    ):
        if sampler is not None:
            shuffle = False
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            sampler=sampler,
        )


TDataset = TypeVar("TDataset", bound=MatchescuDataset)
