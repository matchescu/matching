from unittest.mock import Mock

import numpy as np
import pytest
from matchescu.typing import EntityReference, EntityReferenceIdentifier
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from matchescu.matching.matchers.ml.multiclass.training import (
    AsymmetricMultiClassDataset,
)


@pytest.fixture
def local_tokenizer():
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "COL",
        "VAL",
        "name",
        "city",
        "Acme",
        "NYC",
        "Other",
        "London",
        "first",
        "middle",
        "last",
    ]
    tokenizer = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture
def ref_pair():
    return (
        EntityReference(
            EntityReferenceIdentifier("a", "left"), {"name": "Acme", "city": "NYC"}
        ),
        EntityReference(
            EntityReferenceIdentifier("b", "right"), {"name": "Other", "city": "London"}
        ),
    )


@pytest.fixture
def make_dataset(local_tokenizer, ref_pair):
    def make(pairs=None, tokenizer=None, **kwargs):
        pairs = [ref_pair] if pairs is None else pairs
        split = Mock(
            to_comparison_labels=Mock(
                return_value=(pairs, np.ones(len(pairs), dtype=int))
            )
        )
        return AsymmetricMultiClassDataset(
            Mock(), split, tokenizer or local_tokenizer, **kwargs
        )

    return make
