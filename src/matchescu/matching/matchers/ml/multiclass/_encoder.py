from matchescu.typing import EntityReference
from torch import Tensor, stack


def to_ditto_text(ref: EntityReference, cols: list | None = None) -> str:
    cols = set(cols or [])
    return " ".join(
        f"COL {col} VAL {val}"
        for col, val in ref.as_dict().items()
        if len(cols) == 0 or col in cols
    )


def labels_to_bits(y: Tensor) -> Tensor:
    """Convert published labels to bit targets.

    Published labels have the shape ``(N, )`` and values in ``{0, 1, 2}``.
    The equivalent bit targets have shape ``(N, 2)``. The first bit signals
    whether there's a match in the prescribed input order, the second bit
    signals a match in reverse order. 0 -> [0, *], 1 -> [1, 1], 2 -> [1, 0].
    """
    return stack((y > 0, y == 1), dim=-1).long()
