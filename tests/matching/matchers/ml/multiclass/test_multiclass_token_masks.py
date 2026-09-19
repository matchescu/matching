import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._cross_attention import (
    PooledCrossAttention,
)
from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import ArchitectureType, HeadType

from .._constants import HIDDEN


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("special_col", [False, True])
def test_dataset_preserves_boundary_masks_when_collating(
    make_dataset, ref_pair, local_tokenizer, reverse, special_col
):
    if special_col:
        local_tokenizer.add_special_tokens({"additional_special_tokens": ["COL"]})
    dataset = make_dataset()
    short = make_dataset(left_cols=("name",), right_cols=("city",))

    encoding = dataset._collate([dataset[0], short[0]])[int(reverse)]

    expected = torch.isin(encoding["input_ids"], torch.tensor([0, 2, 3])).long()
    assert torch.equal(encoding["special_tokens_mask"], expected)


@pytest.fixture
def token_batch(local_tokenizer):
    encoding = local_tokenizer(
        ["COL name VAL Acme", "COL name VAL Acme COL city VAL NYC"],
        ["COL name VAL Other", "COL city VAL London"],
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    encoding["col_positions"] = torch.tensor([[1, 6, -1], [1, 5, 10]])
    return encoding


@pytest.mark.parametrize("architecture", list(ArchitectureType))
def test_record_encoding_ignores_boundary_hidden_states(
    token_batch, fake_bert, make_params, architecture
):
    module = MultiClassModule(
        make_params(architecture=architecture, head_type=HeadType.NONE, dropout_p=0.0)
    )
    hidden = torch.randn(*token_batch["input_ids"].shape, HIDDEN)
    fake_bert.return_value = (hidden,)
    expected = module._bert_encode(**token_batch)
    changed = hidden.clone()
    changed[token_batch["special_tokens_mask"].bool()] += 1000
    fake_bert.return_value = (changed,)

    actual = module._bert_encode(**token_batch)

    torch.testing.assert_close(actual, expected)


def test_module_keeps_col_markers_when_pooling(token_batch, fake_bert, make_params):
    module = MultiClassModule(make_params(head_type=HeadType.NONE))
    hidden = torch.zeros(*token_batch["input_ids"].shape, HIDDEN)
    hidden[token_batch["input_ids"] == 4] = 4.0
    fake_bert.return_value = (hidden,)

    encoded = module._bert_encode(**token_batch)

    torch.testing.assert_close(encoded, torch.ones_like(encoded))


def test_forward_consumes_metadata_without_passing_it_to_bert(
    token_batch, fake_bert, make_params
):
    module = MultiClassModule(make_params())
    fake_bert.return_value = (torch.zeros(*token_batch["input_ids"].shape, HIDDEN),)

    logits = module(**token_batch)

    assert (logits.shape, set(fake_bert.call_args.kwargs)) == (
        (2, 3),
        {"attention_mask", "token_type_ids"},
    )


@pytest.mark.parametrize("architecture", list(ArchitectureType))
@pytest.mark.parametrize(
    "empty_a,empty_b", [(True, False), (False, True), (True, True)]
)
def test_record_encoding_is_zero_when_segment_has_only_boundaries(
    local_tokenizer, fake_bert, make_params, architecture, empty_a, empty_b
):
    text = "COL name VAL Acme"
    encoding = local_tokenizer(
        "" if empty_a else text,
        "" if empty_b else text,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    cols = (encoding["input_ids"][0] == 4).nonzero().flatten()
    encoding["col_positions"] = cols.unsqueeze(0)
    module = MultiClassModule(
        make_params(architecture=architecture, head_type=HeadType.NONE, dropout_p=0.0)
    )
    fake_bert.return_value = (torch.ones(*encoding["input_ids"].shape, HIDDEN),)

    encoded = module._bert_encode(**encoding).reshape(2, HIDDEN)

    assert torch.isfinite(encoded).all() and not encoded[[empty_a, empty_b]].any()


@pytest.mark.parametrize(
    "empty_a,empty_b", [(True, False), (False, True), (True, True)]
)
def test_pooled_attention_has_finite_gradients_when_a_batch_row_is_empty(
    empty_a, empty_b
):
    attention = PooledCrossAttention(HIDDEN, dropout=0.0)
    hidden = torch.randn(2, 4, HIDDEN, requires_grad=True)
    mask_a = torch.tensor([[1.0, 1, 0, 0], [0, 0, 0, 0] if empty_a else [1, 1, 0, 0]])
    mask_b = torch.tensor([[0.0, 0, 1, 1], [0, 0, 0, 0] if empty_b else [0, 0, 1, 1]])

    encodings = attention(hidden, mask_a, mask_b)
    sum(encoding.sum() for encoding in encodings).backward()

    assert torch.isfinite(hidden.grad).all()
