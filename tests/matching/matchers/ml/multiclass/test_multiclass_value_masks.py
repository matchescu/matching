from unittest.mock import Mock

import pytest
import torch
from matchescu.typing import EntityReference

from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import ArchitectureType, HeadType

from .._constants import HIDDEN


@pytest.fixture
def attribute_batch(make_dataset, ref_pair):
    def make(values_a, values_b):
        names = ("first", "middle", "last")
        pair = tuple(
            EntityReference(ref.id, dict(zip(names, values)))
            for ref, values in zip(ref_pair, (values_a, values_b))
        )
        dataset = make_dataset(pairs=[pair])
        return dataset._collate([dataset[0]])

    return make


@pytest.mark.parametrize("reverse,expected", [(False, [8, 9, 11]), (True, [11, 8, 9])])
def test_dataset_marks_only_retained_values_when_middle_attributes_are_empty(
    attribute_batch, reverse, expected
):
    encoding = attribute_batch(("Acme", "", "NYC"), ("", "", "London"))[int(reverse)]

    values = encoding["input_ids"][encoding["value_mask"].bool()]

    assert values.tolist() == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_dataset_pads_value_masks_when_pair_lengths_differ(
    make_dataset, ref_pair, reverse
):
    empty = EntityReference(ref_pair[0].id, {"name": "", "city": ""})
    dataset = make_dataset(pairs=[ref_pair, (empty, ref_pair[1])])

    encoding = dataset._collate([dataset[0], dataset[1]])[int(reverse)]

    expected = torch.isin(encoding["input_ids"], torch.tensor([8, 9, 10, 11]))
    assert torch.equal(encoding["value_mask"].bool(), expected)


@pytest.mark.parametrize("value", [None, "None", "null", "[UNK]", "VAL"])
def test_dataset_keeps_value_tokens_when_text_looks_like_a_null(
    make_dataset, ref_pair, value
):
    pair = (EntityReference(ref_pair[0].id, {"name": value}), ref_pair[1])
    encoding = make_dataset(pairs=[pair])[0][0]

    selected = encoding["value_mask"].bool() & (encoding["token_type_ids"] == 0)

    assert selected.sum().item() == 1


@pytest.fixture
def per_attribute_module(make_params):
    return MultiClassModule(
        make_params(
            architecture=ArchitectureType.BERT_PER_ATTR_CROSS_ATTN,
            head_type=HeadType.NONE,
            dropout_p=0.0,
        )
    )


@pytest.mark.parametrize(
    "values_a,values_b,paired_ordinals",
    [
        (("Acme", "", "NYC"), ("Other", "London", "London"), [0, 2]),
        (("Acme", "NYC", "NYC"), ("Other", "", "London"), [0, 2]),
        (("", "NYC", "NYC"), ("Other", "London", ""), [1]),
        (("Acme", "", ""), ("", "London", ""), []),
        (("", "", ""), ("", "", ""), []),
    ],
)
def test_per_attribute_attention_preserves_ordinals_when_values_are_empty(
    attribute_batch,
    per_attribute_module,
    fake_bert,
    monkeypatch,
    values_a,
    values_b,
    paired_ordinals,
):
    encoding = attribute_batch(values_a, values_b)[0]
    hidden = torch.arange(encoding["input_ids"].numel()).float()[None, :, None]
    fake_bert.return_value = (hidden.expand(-1, -1, HIDDEN),)
    attention = per_attribute_module.cross_attention.attn_a
    spy = Mock(wraps=attention.forward)
    monkeypatch.setattr(attention, "forward", spy)

    per_attribute_module._bert_encode(**encoding)

    cols = encoding["col_positions"][0].tolist()
    actual = [
        (call.kwargs["key"][0, 0, 0].item(), call.kwargs["query"][0, 0, 0].item())
        for call in spy.call_args_list
    ]
    assert actual == [(cols[i], cols[i + 3]) for i in paired_ordinals]


@pytest.mark.parametrize(
    "values_a,values_b",
    [(("", "", ""), ("", "", "")), (("Acme", "", ""), ("", "London", ""))],
)
def test_per_attribute_encoding_is_zero_when_no_attribute_pair_has_values(
    attribute_batch, per_attribute_module, fake_bert, values_a, values_b
):
    encoding = attribute_batch(values_a, values_b)[0]
    fake_bert.return_value = (torch.ones(*encoding["input_ids"].shape, HIDDEN),)

    actual = per_attribute_module._bert_encode(**encoding)

    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_forward_consumes_value_metadata_without_passing_it_to_bert(
    attribute_batch, per_attribute_module, fake_bert
):
    encoding = attribute_batch(("", "", ""), ("", "", ""))[0]
    encoding["value_mask"] = torch.zeros_like(encoding["input_ids"], dtype=torch.bool)
    fake_bert.return_value = (torch.ones(*encoding["input_ids"].shape, HIDDEN),)

    logits = per_attribute_module(**encoding)

    assert (
        logits.shape,
        torch.isfinite(logits).all().item(),
        set(fake_bert.call_args.kwargs),
    ) == ((1, 3), True, {"attention_mask", "token_type_ids"})


def test_per_attribute_attention_keeps_populated_behavior_with_value_metadata(
    attribute_batch, per_attribute_module, fake_bert
):
    encoding = attribute_batch(("Acme", "NYC", "NYC"), ("Other", "London", "London"))[0]
    fake_bert.return_value = (torch.randn(*encoding["input_ids"].shape, HIDDEN),)
    actual = per_attribute_module._bert_encode(**encoding)
    encoding.pop("value_mask")

    expected = per_attribute_module._bert_encode(**encoding)

    torch.testing.assert_close(actual, expected)


def test_dataset_marks_no_values_when_truncation_retains_only_schema(make_dataset):
    dataset = make_dataset(left_cols=("name",), right_cols=("name",), max_len=9)

    forward, reverse, _ = dataset[0]

    assert not forward["value_mask"].any() and not reverse["value_mask"].any()
