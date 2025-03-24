from matchescu.typing import EntityReference


def to_text(ref: EntityReference, cols: list = None) -> str:
    cols = set(cols or [])
    return " ".join(
        f"COL {col} VAL {val}"
        for col, val in ref.as_dict().items()
        if len(cols) == 0 or col in cols
    )
