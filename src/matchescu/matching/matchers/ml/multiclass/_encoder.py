import torch
from matchescu.typing import EntityReference


def to_ditto_text(ref: EntityReference, cols: list | None = None) -> str:
    cols = set(cols or [])
    return " ".join(
        f"COL {col} VAL {val}"
        for col, val in ref.as_dict().items()
        if len(cols) == 0 or col in cols
    )


def value_token_mask(
    encoding: dict[str, torch.Tensor], col_token_id: int, val_token_id: int
) -> torch.Tensor:
    """Mark retained value content without interpreting textual null values.

    :returns: A token mask excluding schema, boundaries, and padding.
    """
    ids = encoding["input_ids"]
    special = encoding["special_tokens_mask"].bool()
    positions = torch.arange(ids.size(-1), device=ids.device)
    starts = torch.where((ids == col_token_id) | special, positions, -1)
    starts = starts.cummax(dim=-1).values
    vals = torch.where(ids == val_token_id, positions, -1)
    preceding_vals = torch.nn.functional.pad(vals[..., :-1], (1, 0), value=-1)
    preceding_vals = preceding_vals.cummax(dim=-1).values
    owns_span = ids.gather(-1, starts.clamp(min=0)) == col_token_id
    return (
        owns_span
        & (preceding_vals > starts)
        & ~special
        & encoding["attention_mask"].bool()
    )
