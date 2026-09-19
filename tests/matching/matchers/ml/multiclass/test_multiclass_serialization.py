from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "left_cols,right_cols,left_text,right_text",
    [
        (("name",), ("city",), "COL name VAL Acme", "COL city VAL London"),
        (None, ("city",), "COL name VAL Acme COL city VAL NYC", "COL city VAL London"),
        (
            ("name",),
            None,
            "COL name VAL Acme",
            "COL name VAL Other COL city VAL London",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_dataset_serializes_selected_columns_when_pair_is_reversed(
    make_dataset, local_tokenizer, left_cols, right_cols, left_text, right_text, reverse
):
    tokenizer = Mock(wraps=local_tokenizer)
    dataset = make_dataset(
        tokenizer=tokenizer, left_cols=left_cols, right_cols=right_cols
    )
    tokenizer.reset_mock()

    dataset[0]

    call = tokenizer.call_args_list[int(reverse)]
    expected = (right_text, left_text) if reverse else (left_text, right_text)
    assert (call.kwargs["text"], call.kwargs["text_pair"]) == expected
