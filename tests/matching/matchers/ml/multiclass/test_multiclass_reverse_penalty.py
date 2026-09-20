import pytest


def test_compute_loss_when_penalty_zero_adds_nothing(
    trainer, loss_fn, logits, logits_rev, targets, targets_rev, default_loss
):
    loss = trainer._compute_loss(0, loss_fn, [logits, logits_rev, targets, targets_rev])
    assert loss.item() == default_loss


@pytest.mark.parametrize(
    "trainer,multiplier", [(1.0, 1.0), (2.0, 2.0)], indirect=["trainer"]
)
def test_compute_loss_penalty_scales_linearly_with_weight(
    trainer, loss_fn, logits, logits_rev, targets, targets_rev, multiplier, default_loss
):
    penalty = trainer._directional_margin_loss(
        logits, logits_rev, targets, trainer._DIRECTIONAL_MARGIN
    ).item()
    expected = default_loss + penalty * multiplier

    actual = trainer._compute_loss(
        0, loss_fn, [logits, logits_rev, targets, targets_rev]
    ).item()

    assert expected == actual
