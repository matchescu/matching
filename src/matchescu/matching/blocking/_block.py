from dataclasses import dataclass, field
from typing import Any

from matchescu.typing import EntityReference


@dataclass
class Block:
    key: Any = field(init=True, repr=True, hash=True, compare=True)
    references: dict[str, list[EntityReference]] = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def add_reference(self, source_name: str, reference: EntityReference) -> "Block":
        refs = self.references.get(source_name) or []
        refs.append(reference)
        self.references[source_name] = refs
        return self
