from matchescu.typing import EntityReference


def to_text(ref: EntityReference) -> str:
    return " ".join(f"COL {col} VAL {val}" for col, val in ref.as_dict().items())
