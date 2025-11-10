import itertools
import random
import re
from collections.abc import Sequence, Generator
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.ditto._ref_adapter import to_text
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier


def alphanumeric(token):
    return "".join([ch if ch.isalnum() else " " for ch in token])


class Augmenter(object):
    """Data augmentation operator.

    Support both span and attribute level augmentation operators.
    """

    def __init__(self):
        self._augmentations: dict[
            str, Callable[[Sequence, Sequence], tuple[Sequence, Sequence]]
        ] = {
            "del": self._del,
            "swap": self._swap,
            "drop_len": self._drop_len,
            "drop_sym": self._drop_symbols,
            "drop_same": self._drop_same,
            "drop_token": self._drop_token,
            "ins": self._insert,
            "append_col": self._append_col,
            "drop_col": self._drop_col,
        }

    def augment(self, tokens: Sequence[str], labels: Sequence[str], op="del"):
        augmentation = self._augmentations.get(op, lambda x, y: (x, y))
        return augmentation(labels, tokens)

    @staticmethod
    def _drop_col(labels, tokens):
        col_starts = [i for i in range(len(tokens)) if tokens[i] == "COL"]
        col_ends = [0] * len(col_starts)
        col_lens = [0] * len(col_starts)
        for i, pos in enumerate(col_starts):
            if i == len(col_starts) - 1:
                col_lens[i] = len(tokens) - pos
                col_ends[i] = len(tokens) - 1
            else:
                col_lens[i] = col_starts[i + 1] - pos
                col_ends[i] = col_starts[i + 1] - 1

            if tokens[col_ends[i]] == "[SEP]":
                col_ends[i] -= 1
                col_lens[i] -= 1
        candidates = [i for i, le in enumerate(col_lens) if le <= 8]
        if len(candidates) > 0:
            idx = random.choice(candidates)
            start, end = col_starts[idx], col_ends[idx]
            new_tokens = tokens[:start] + tokens[end + 1 :]
            new_labels = labels[:start] + labels[end + 1 :]
        else:
            new_tokens, new_labels = tokens, labels
        return new_labels, new_tokens

    @staticmethod
    def _append_col(labels, tokens):
        col_starts = [i for i in range(len(tokens)) if tokens[i] == "COL"]
        col_ends = [0] * len(col_starts)
        col_lens = [0] * len(col_starts)
        for i, pos in enumerate(col_starts):
            if i == len(col_starts) - 1:
                col_lens[i] = len(tokens) - pos
                col_ends[i] = len(tokens) - 1
            else:
                col_lens[i] = col_starts[i + 1] - pos
                col_ends[i] = col_starts[i + 1] - 1

            if tokens[col_ends[i]] == "[SEP]":
                col_ends[i] -= 1
                col_lens[i] -= 1
                break
        candidates = [i for i, le in enumerate(col_lens) if le > 0]
        if len(candidates) < 2:
            return tokens, labels
        else:
            idx1, idx2 = random.sample(candidates, k=2)
            start1, end1 = col_starts[idx1], col_ends[idx1]
            sub_tokens = tokens[start1 : end1 + 1]
            sub_labels = labels[start1 : end1 + 1]
            val_pos = 0
            for i, token in enumerate(sub_tokens):
                if token == "VAL":
                    val_pos = i + 1
                    break
            sub_tokens = sub_tokens[val_pos:]
            sub_labels = sub_labels[val_pos:]

            end2 = col_ends[idx2]
            new_tokens = []
            new_labels = []
            for i in range(len(tokens)):
                if start1 <= i <= end1:
                    continue
                new_tokens.append(tokens[i])
                new_labels.append(labels[i])
                if i == end2:
                    new_tokens += sub_tokens
                    new_labels += sub_labels
            return new_tokens, new_labels

    def _insert(self, labels, tokens):
        pos = self.sample_position(labels)
        symbol = random.choice("-*.,#&")
        new_tokens = tokens[:pos] + [symbol] + tokens[pos:]
        new_labels = labels[:pos] + ["O"] + labels[pos:]
        return new_tokens, new_labels

    @staticmethod
    def _drop_token(labels, tokens):
        new_tokens, new_labels = [], []
        for token, label in zip(tokens, labels):
            if label != "O" or random.randint(0, 4) != 0:
                new_tokens.append(token)
                new_labels.append(label)
        return new_labels, new_tokens

    @staticmethod
    def _drop_same(labels, tokens):
        left_token = set([])
        right_token = set([])
        left = True
        for token, label in zip(tokens, labels):
            if label == "O":
                token = token.lower()
                if left:
                    left_token.add(token)
                else:
                    right_token.add(token)
            if token == "[SEP]":
                left = False
        same = left_token & right_token
        targets = random.choices(list(same), k=1)
        new_tokens, new_labels = [], []
        for token, label in zip(tokens, labels):
            if token.lower() not in targets or label != "O":
                new_tokens.append(token)
                new_labels.append(label)
        return new_labels, new_tokens

    @staticmethod
    def _drop_symbols(labels, tokens):
        dropped_tokens = [alphanumeric(token) for token in tokens]
        new_tokens = []
        new_labels = []
        for token, d_token, label in zip(tokens, dropped_tokens, labels):
            if random.randint(0, 4) != 0 or label != "O":
                new_tokens.append(token)
                new_labels.append(label)
            else:
                if d_token != "":
                    new_tokens.append(d_token)
                    new_labels.append(label)
        return new_labels, new_tokens

    def _swap(self, labels, tokens):
        span_len = random.randint(2, 4)
        pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
        if pos1 < 0:
            return tokens, labels
        sub_arr = tokens[pos1 : pos2 + 1]
        random.shuffle(sub_arr)
        new_tokens = tokens[:pos1] + sub_arr + tokens[pos2 + 1 :]
        new_labels = tokens[:pos1] + ["O"] * (pos2 - pos1 + 1) + labels[pos2 + 1 :]
        return new_tokens, new_labels

    def _del(self, labels, tokens):
        span_len = random.randint(1, 2)
        pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
        if pos1 < 0:
            return tokens, labels
        new_tokens = tokens[:pos1] + tokens[pos2 + 1 :]
        new_labels = tokens[:pos1] + labels[pos2 + 1 :]
        return new_tokens, new_labels

    @staticmethod
    def _drop_len(labels, tokens):
        all_lens = [len(token) for token, label in zip(tokens, labels) if label == "O"]
        if len(all_lens) == 0:
            return tokens, labels
        target_lens = random.choices(all_lens, k=1)
        new_tokens = []
        new_labels = []

        for token, label in zip(tokens, labels):
            if label != "O" or len(token) not in target_lens:
                new_tokens.append(token)
                new_labels.append(label)
        return new_tokens, new_labels

    def augment_sent(self, text, op="all"):
        """Performs data augmentation on a classification example.

        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.

        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            str: the augmented sentence
        """
        # 50% of chance of flipping
        if " [SEP] " in text and random.randint(0, 1) == 0:
            left, right = text.split(" [SEP] ")
            text = right + " [SEP] " + left

        # tokenize the sentence
        tokens = text.split(" ")

        # avoid the special tokens
        labels = []
        for token in tokens:
            if token in ["COL", "VAL"]:
                labels.append("HD")
            elif token in ["[CLS]", "[SEP]"]:
                labels.append("<SEP>")
            else:
                labels.append("O")

        if op == "all":
            # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
            N = 3
            ops = ["del", "swap", "drop_col", "append_col"]
            for op in random.choices(ops, k=N):
                tokens, labels = self.augment(tokens, labels, op=op)
        else:
            tokens, labels = self.augment(tokens, labels, op=op)
        results = " ".join(tokens)
        return results

    @staticmethod
    def _process_index(
        idx: int, span_len: int, labels: Sequence[str]
    ) -> Generator[tuple[int, int]]:
        span_end = idx + span_len
        if span_end - 1 >= len(labels):
            return
        span_text = "".join(labels[idx:span_end])
        pattern = "^O{%d}$" % span_len
        if not re.match(pattern, span_text, re.S):
            return
        yield idx, span_end - 1

    def _enumerate_spans(
        self, tokens: Sequence[str], labels: Sequence[str], span_len: int = 3
    ) -> Generator[tuple[int, int]]:
        yield from (
            t
            for idx, _ in enumerate(tokens)
            for t in self._process_index(idx, span_len, labels)
        )

    def sample_span(
        self, tokens: Sequence[str], labels: Sequence[str], span_len: int = 3
    ) -> tuple[int, int]:
        candidates = list(self._enumerate_spans(tokens, labels, span_len))
        if len(candidates) == 0:
            return -1, -1
        return random.choice(candidates)

    @staticmethod
    def sample_position(labels):
        candidates = list(
            map(lambda x: x[0], filter(lambda x: x[1] == "O", enumerate(labels)))
        )
        return random.choice(candidates) if len(candidates) > 0 else -1


class DittoDataset(Dataset):
    """EM dataset"""

    def __init__(
        self,
        id_table: IdTable,
        comparison_space: BinaryComparisonSpace,
        ground_truth: set[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]],
        tokenizer: PreTrainedTokenizerFast,
        max_len=256,
        size=None,
        augmentations=None,
        left_cols: tuple | None = None,
        right_cols: tuple | None = None,
    ):
        self.__id_table = id_table
        self.__pairs = list(
            map(lambda pair: tuple(self.__id_table.get_all(pair)), comparison_space)
        )
        self.__labels = list(
            map(lambda pair: int(pair in ground_truth), comparison_space)
        )
        self.__comparison_space = comparison_space
        self.__ground_truth = ground_truth
        self.__tokenizer = tokenizer
        self.__max_len = max_len
        self.__size = size
        self.__left_cols = left_cols
        self.__right_cols = right_cols
        self.da = augmentations
        if augmentations is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.__pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left, right = self.__pairs[idx]
        left_text = to_text(left, self.__left_cols)
        right_text = to_text(right, self.__right_cols)
        x = self.__tokenizer.encode(
            text=left_text,
            text_pair=right_text,
            max_length=self.__max_len,
            truncation=True,
        )

        # augment if da is set
        if self.da is not None:
            augmented = self.augmenter.augment_sent(
                f"{left_text} [SEP] {right_text}", self.da
            )
            left_aug, right_aug = augmented.split(" [SEP] ")
            x_aug = self.__tokenizer.encode(
                text=left_aug,
                text_pair=right_aug,
                max_length=self.__max_len,
                truncation=True,
            )
            return x, x_aug, self.__labels[idx]
        else:
            return x, self.__labels[idx]

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

    @staticmethod
    def __zero_padded(batch: list[tuple]) -> tuple[torch.LongTensor, ...]:
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            n = max(map(len, x1 + x2))
            x1 = DittoDataset.__pad(x1, n)
            x2 = DittoDataset.__pad(x2, n)
            return x1, x2, torch.LongTensor(y)
        else:
            x, y = zip(*batch)
            n = max(map(len, x))
            x = DittoDataset.__pad(x, n)
            return x, torch.LongTensor(y)

    def get_data_loader(
        self, batch_size: int = 32, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=DittoDataset.__zero_padded,
        )
