from unittest.mock import Mock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._cross_attention import (
    PooledCrossAttention,
)


@pytest.fixture
def attention():
    attention = PooledCrossAttention(2, num_heads=1, dropout=0.0)
    with torch.no_grad():
        attention.attn.in_proj_weight.copy_(
            torch.cat(
                [torch.eye(2), torch.tensor([[0.0, 2.0], [1.0, 0.0]]), torch.eye(2)]
            )
        )
        attention.attn.in_proj_bias.fill_(0.25)
        attention.attn.out_proj.weight.copy_(torch.eye(2))
        attention.attn.out_proj.bias.fill_(0.5)
    return attention


@pytest.fixture
def segments():
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [9.0, 9.0]]]
    )
    mask_a = torch.tensor([[1.0, 0, 0, 0, 0]])
    mask_b = torch.tensor([[0.0, 1, 1, 1, 0]])
    return hidden, mask_a, mask_b


@pytest.mark.parametrize("direction", [0, 1], ids=["enc_a", "enc_b"])
def test_encoding_changes_when_opposite_segment_tokens_are_perturbed(
    attention, segments, direction
):
    hidden, mask_a, mask_b = segments
    opposite_mask = (mask_b, mask_a)[direction]
    expected = attention(hidden, mask_a, mask_b)[direction]
    changed = hidden.clone()
    changed[opposite_mask.bool()] += torch.tensor([3.0, -1.0])

    actual = attention(changed, mask_a, mask_b)[direction]

    assert not torch.allclose(actual, expected)


@pytest.mark.parametrize("direction", [0, 1], ids=["enc_a", "enc_b"])
def test_attention_uses_pooled_query_with_opposite_keys_values_and_mask(
    attention, segments, monkeypatch, direction
):
    hidden, mask_a, mask_b = segments
    query_mask, context_mask = (mask_a, mask_b) if direction == 0 else (mask_b, mask_a)
    spy = Mock(wraps=attention.attn.forward)
    monkeypatch.setattr(attention.attn, "forward", spy)

    attention(hidden, mask_a, mask_b)

    arguments = spy.call_args_list[direction].kwargs
    pooled = (hidden * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(
        1, keepdim=True
    )
    context = hidden * context_mask.unsqueeze(-1)
    torch.testing.assert_close(arguments["query"], pooled.unsqueeze(1))
    torch.testing.assert_close(arguments["key"], context)
    torch.testing.assert_close(arguments["value"], context)
    torch.testing.assert_close(arguments["key_padding_mask"], context_mask == 0)


@pytest.mark.parametrize("direction", [0, 1], ids=["enc_a", "enc_b"])
def test_padding_mask_matches_attention_to_only_opposite_tokens(
    attention, segments, direction
):
    hidden, mask_a, mask_b = segments
    query_mask, context_mask = (mask_a, mask_b) if direction == 0 else (mask_b, mask_a)
    query = hidden[query_mask.bool()].mean(0).reshape(1, 1, 2)
    context = hidden[context_mask.bool()].unsqueeze(0)
    expected, _ = attention.attn(query, context, context)

    actual = attention(hidden, mask_a, mask_b)[direction]

    torch.testing.assert_close(actual, expected.squeeze(1))


@pytest.mark.parametrize(
    "empty_a,empty_b", [(True, False), (False, True), (True, True)]
)
def test_encodings_are_zero_when_query_or_opposite_context_is_empty(
    attention, segments, empty_a, empty_b
):
    hidden, mask_a, mask_b = segments
    hidden, mask_a, mask_b = (
        hidden.repeat(2, 1, 1),
        mask_a.repeat(2, 1),
        mask_b.repeat(2, 1),
    )
    healthy = attention(hidden, mask_a, mask_b)
    if empty_a:
        mask_a[1] = 0
    if empty_b:
        mask_b[1] = 0

    encodings = attention(hidden, mask_a, mask_b)

    for actual, expected in zip(encodings, healthy):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))


@pytest.mark.parametrize(
    "empty_a,empty_b", [(True, False), (False, True), (True, True)]
)
def test_attention_parameters_have_finite_gradients_when_context_is_empty(
    attention, segments, empty_a, empty_b
):
    hidden, mask_a, mask_b = segments
    hidden.requires_grad_()
    if empty_a:
        mask_a.zero_()
    if empty_b:
        mask_b.zero_()

    sum(encoding.sum() for encoding in attention(hidden, mask_a, mask_b)).backward()

    for tensor in (hidden, *attention.parameters()):
        assert torch.isfinite(tensor.grad).all()


def test_shared_attention_pair_outputs_swap_when_segment_masks_swap(
    attention, segments
):
    hidden, mask_a, mask_b = segments
    forward_a, forward_b = attention(hidden, mask_a, mask_b)

    reverse_a, reverse_b = attention(hidden, mask_b, mask_a)

    torch.testing.assert_close(forward_a, reverse_b)
    torch.testing.assert_close(forward_b, reverse_a)
