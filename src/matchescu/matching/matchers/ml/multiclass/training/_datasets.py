import numpy as np
import torch
from matchescu.reference_store.id_table import IdTable
from transformers import BatchEncoding, PreTrainedTokenizerFast

from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.matching.matchers.ml.torch import set_random_seed

from ...training import MatchescuDataset
from .._encoder import to_ditto_text, value_token_mask


class AsymmetricMultiClassDataset(MatchescuDataset):
    _COL_TOKEN = "COL"

    def __init__(
        self,
        id_table: IdTable,
        split: Split,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 256,
        left_cols: tuple | None = None,
        right_cols: tuple | None = None,
        random_seed: int | None = 42,
    ):
        super().__init__(id_table, split)
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__left_cols = left_cols
        self.__right_cols = right_cols
        self.__label_counts = np.bincount(self._labels)
        self.__col_token_id = tokenizer(self._COL_TOKEN, add_special_tokens=False)[
            "input_ids"
        ][0]
        self.__val_token_id = tokenizer("VAL", add_special_tokens=False)["input_ids"][0]
        if random_seed is not None:
            set_random_seed(random_seed)

    @property
    def label_counts(self) -> np.ndarray:
        return self.__label_counts

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            encodings, attention masks, token type ids, and COL token
            positions for both normal and reversed input orders.
        """
        left, right = self._pairs[idx]
        left_text = to_ditto_text(left, self.__left_cols)
        right_text = to_ditto_text(right, self.__right_cols)

        y = self._labels[idx]

        x_fwd = {
            k: v.squeeze(0)
            for k, v in self.__tokenizer(
                text=left_text,
                text_pair=right_text,
                max_length=self.__max_len,
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
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
                return_special_tokens_mask=True,
            ).items()
        }

        x_fwd["col_positions"] = self._find_col_positions(x_fwd["input_ids"])
        x_rev["col_positions"] = self._find_col_positions(x_rev["input_ids"])
        for encoding in (x_fwd, x_rev):
            encoding["value_mask"] = value_token_mask(
                encoding, self.__col_token_id, self.__val_token_id
            )

        return x_fwd, x_rev, y

    def _find_col_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = (input_ids == self.__col_token_id).nonzero(as_tuple=False)
        return (
            positions.squeeze(-1)
            if positions.numel() > 0
            else torch.tensor([-1], dtype=torch.long)
        )

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

    def _pad_col_positions(
        self, col_positions_list: list[torch.Tensor]
    ) -> torch.Tensor:
        max_cols = max(t.size(0) for t in col_positions_list)
        padded = torch.full(
            (len(col_positions_list), max_cols),
            -1,
            dtype=torch.long,
        )
        for i, t in enumerate(col_positions_list):
            padded[i, : t.size(0)] = t
        return padded

    def _collate(self, batch: list[tuple]) -> tuple[dict, dict, torch.LongTensor]:
        x_fwd_list, x_rev_list, y = zip(*batch)

        fwd_cols = [item.pop("col_positions") for item in x_fwd_list]
        rev_cols = [item.pop("col_positions") for item in x_rev_list]
        fwd_values = [item.pop("value_mask") for item in x_fwd_list]
        rev_values = [item.pop("value_mask") for item in x_rev_list]

        x_fwd_padded = self._pad(x_fwd_list)
        x_rev_padded = self._pad(x_rev_list)

        x_fwd_padded["col_positions"] = self._pad_col_positions(fwd_cols)
        x_rev_padded["col_positions"] = self._pad_col_positions(rev_cols)
        for encoding, values in (
            (x_fwd_padded, fwd_values),
            (x_rev_padded, rev_values),
        ):
            encoding["value_mask"] = torch.nn.utils.rnn.pad_sequence(
                values,
                batch_first=True,
                padding_value=False,
                padding_side=self.__tokenizer.padding_side,
            )

        return x_fwd_padded, x_rev_padded, torch.tensor(y, dtype=torch.int64)
