from typing import Iterable, Iterator

from matchescu.typing import Trait, Record


class ListDataSource:
    def __init__(self, name: str, traits: Iterable[Trait]):
        self.name = name
        self.traits = traits
        self._items: list[tuple] = []

    def append(self, item: Record) -> "ListDataSource":
        if item is None:
            return self

        if isinstance(item, dict):
            self._items.append(tuple(item.values()))
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            self._items.append(tuple(item))
        else:
            self._items.append((item,))

        return self

    def extend(self, items: Iterable[Record]) -> "ListDataSource":
        for item in items:
            self.append(item)
        return self

    def __iter__(self) -> Iterator[Record]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
