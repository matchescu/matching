import torch
from typing import Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding

from matchescu.data import Record
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.matching.matchers.ml.training import MatchescuDataset
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier

from matchescu.matching.matchers.ml.deeper._encoder import (
    to_deeper_repr,
    ensure_attr_map,
)


class DeepERDataset(MatchescuDataset):
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
        super().__init__(id_table, split)
        self.__tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            DeepERDataset.__DEFAULT_TOKENIZER_NAME,
        )
        self.__max_len = max_len
        left, right = next(iter(self._pairs))
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
        return len(self._labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[list[BatchEncoding], list[BatchEncoding], torch.Tensor]:
        left, right = self._pairs[idx]
        left_enc, right_enc = to_deeper_repr(
            left, right, self.__tokenizer, self.__attr_map, self.__max_len
        )
        labels = torch.tensor(self._labels[idx], dtype=torch.float)
        return left_enc, right_enc, labels

    def _collate(
        self, batch: list[tuple[list[BatchEncoding], list[BatchEncoding], torch.Tensor]]
    ) -> tuple[
        list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], torch.Tensor
    ]:
        left_list, right_list, labels = zip(*batch)
        left_batch, right_batch = [], []
        for i in range(self.__attr_count):
            left_batch.append(
                {
                    "input_ids": torch.cat(
                        [left[i]["input_ids"] for left in left_list]
                    ),
                    "attention_mask": torch.cat(
                        [left[i]["attention_mask"] for left in left_list]
                    ),
                }
            )
            right_batch.append(
                {
                    "input_ids": torch.cat(
                        [right[i]["input_ids"] for right in right_list]
                    ),
                    "attention_mask": torch.cat(
                        [right[i]["attention_mask"] for right in right_list]
                    ),
                }
            )
        labels = torch.stack(labels)
        return left_batch, right_batch, labels


if __name__ == "__main__":
    from matchescu.matching.evaluation.data.benchmark import (
        MagellanBenchmarkData,
        MagellanTraits,
    )

    traits = MagellanTraits()["beer"]

    data = MagellanBenchmarkData("data/magellan/beer")

    def _id_factory(rows: Iterable[Record], source: str) -> EntityReferenceIdentifier:
        return EntityReferenceIdentifier(next(iter(rows))["id"], source)

    data.load_left(traits)
    data.load_right(traits)
    data.load_splits()
    dataset = DeepERDataset(data.id_table, data.train_split)
    for item in dataset.get_data_loader():
        print(item)
