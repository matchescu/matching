"""The DeepMatcher architecture works with pairs of entity references. The
corresponding attributes (given by the user) are always encoded pairwise using
a tokenizer of choice. Our implementation uses ``PreTrainedTokenizerBase`` from
the ``transformers`` library.
"""

import torch
from transformers import PreTrainedTokenizerBase

from matchescu.typing import EntityReference


def ensure_attr_map(
    a: EntityReference,
    b: EntityReference,
    attr_map: dict | None = None,
    excluded: set[str] | None = None,
):
    excluded = excluded or set()
    min_attr_count = min(len(a), len(b))
    attr_map = attr_map or {i: i for i in range(min_attr_count)}
    return {
        k: v for k, v in attr_map.items() if k not in excluded and v not in excluded
    }


def to_deepmatcher_repr(
    a: EntityReference,
    b: EntityReference,
    tokenizer: PreTrainedTokenizerBase,
    attr_map: dict,
    max_len: int = 30,
) -> dict[str, torch.Tensor]:
    left_tokens, right_tokens = [], []
    for left_key, right_key in attr_map.items():
        left_text = str(a[left_key])
        left_enc = tokenizer(
            left_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        right_text = str(b[right_key])
        right_enc = tokenizer(
            right_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        left_tokens.append(left_enc["input_ids"])
        right_tokens.append(right_enc["input_ids"])
    left_attrs = torch.tensor(left_tokens, dtype=torch.long)
    right_attrs = torch.tensor(right_tokens, dtype=torch.long)
    return {"left_attrs": left_attrs, "right_attrs": right_attrs}
