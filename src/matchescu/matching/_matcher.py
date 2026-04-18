from typing import Protocol, TypeVar

from matchescu.typing import EntityReference
from matchescu.similarity import MatchResult

TRef = TypeVar("TRef", bound=EntityReference)


class Matcher[TRef](Protocol):
    def __call__(self, left: TRef, right: TRef) -> MatchResult:
        """Return a similarity score between ``left`` and ``right``.

        :param left: an entity reference of any kind
        :param right: an entity reference of any kind

        :return: a ``float`` value ranged between 0 and 1 which represents the
            probability that ``left`` matches ``right``.
        """
        raise NotImplementedError()
