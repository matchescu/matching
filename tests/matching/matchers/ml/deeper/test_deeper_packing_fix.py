"""Regression tests for the ``__compose_attr`` packing-reorder fix.

``pack_padded_sequence(enforce_sorted=False)`` silently reorders the batch by
descending length. ``__compose_attr`` must restore the original batch order
via ``packed.unsorted_indices`` before returning the hidden state. Without
that, the left and right representations of a pair get paired with the wrong
examples, corrupting ``diff = |h_l - h_r|``.
"""

import torch

from .._constants import BATCH, SEQ


def _varied_lengths_batch():
    """A 4-row batch whose lengths [2, 6, 1, 4] sort non-trivially.

    The descending-length permutation is ``[1, 3, 0, 2]`` (not identity), so
    a missing ``unsorted_indices`` would visibly permute the output rows.
    """
    mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
    return ids, mask


def test_compose_attr_restores_original_batch_order_when_lengths_vary(
    deeper_module, row_signal_lstm
):
    """``__compose_attr`` must return rows in the original batch order.

    The ``row_signal_lstm`` mock returns ``h_n`` in the pack's sorted order,
    where each row carries its *original* index as a constant signal. A
    correct unsort yields ``[0, 1, 2, 3]``; the pre-fix code returned the
    sorted order ``[1, 3, 0, 2]``.
    """
    deeper_module._lstm = row_signal_lstm

    ids, mask = _varied_lengths_batch()
    emb = deeper_module._DeepERModule__encode_all_attrs(
        [{"input_ids": ids, "attention_mask": mask}]
    )[0]
    h = deeper_module._DeepERModule__compose_attr(emb, mask)

    signals = h[:, 0].tolist()
    assert signals == [
        0.0,
        1.0,
        2.0,
        3.0,
    ], f"__compose_attr did not restore original batch order; got {signals}"
