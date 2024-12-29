import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Callable, Iterable, Hashable


@dataclass
class Block:
    __DEFAULT_SOURCE = "DEFAULT"

    key: Any = field(init=True, repr=True, hash=True, compare=True)
    references: dict[str, list[Hashable]] = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self):
        if self.key is None or not self.key.strip():
            raise ValueError("invalid blocking key")

    def __update_references(
        self,
        param: Any,
        source_name: str | None,
        op: Callable[[list], Callable[[Any], None]],
    ) -> None:
        src = (
            source_name
            if source_name is not None and source_name.strip()
            else self.__DEFAULT_SOURCE
        )
        refs = self.references.get(src) or []
        op(refs)(param)
        self.references[src] = refs

    def append(self, ref_id: Hashable, source_name: str | None = None) -> "Block":
        self.__update_references(ref_id, source_name, lambda x: x.append)
        return self

    def extend(
        self, ref_ids: Iterable[Hashable], source_name: str | None = None
    ) -> "Block":
        self.__update_references(ref_ids, source_name, lambda x: x.extend)
        return self

    def candidate_pairs(
        self, generate_deduplication_pairs: bool = True
    ) -> Iterator[tuple[Hashable, Hashable]]:
        n_sources = len(self.references)
        if n_sources < 1:
            yield from ()
        elif n_sources < 2 and generate_deduplication_pairs:
            for ref_ids in self.references.values():
                for pair in itertools.combinations(ref_ids, 2):
                    yield pair
        else:
            sources = list(self.references.keys())
            for a, b in itertools.combinations(sources, 2):
                for prod in itertools.product(self.references[a], self.references[b]):
                    yield prod
