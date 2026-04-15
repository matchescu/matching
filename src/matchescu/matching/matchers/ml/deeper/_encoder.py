"""The DeepMatcher architecture works with pairs of entity references. The
corresponding attributes (given by the user) are always encoded pairwise using
a tokenizer of choice. Our implementation uses ``PreTrainedTokenizerBase`` from
the ``transformers`` library.
"""

from transformers import PreTrainedTokenizerBase, BatchEncoding

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


def to_deeper_repr(
    a: EntityReference,
    b: EntityReference,
    tokenizer: PreTrainedTokenizerBase,
    attr_map: dict,
    max_len: int = 30,
) -> tuple[list[BatchEncoding], list[BatchEncoding]]:
    left_tokens, right_tokens = [], []
    for left_key, right_key in attr_map.items():
        left_text = str(a[left_key])
        left_enc = tokenizer(
            left_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        right_text = str(b[right_key])
        right_enc = tokenizer(
            right_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        left_tokens.append(left_enc)
        right_tokens.append(right_enc)
    return left_tokens, right_tokens
