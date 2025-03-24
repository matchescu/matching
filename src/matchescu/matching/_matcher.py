from typing import Protocol

from matchescu.typing import EntityReference


class Matcher(Protocol):
    @property
    def non_match_threshold(self) -> float:
        """Similarity score (ranged 0..1) below which two references are considered to truly mismatch."""
        pass

    @property
    def match_threshold(self) -> float:
        """Similarity score (ranged 0..1) above which two references are considered to truly match one another."""
        pass

    def __call__(self, left: EntityReference, right: EntityReference) -> float:
        """Return a similarity score between ``left`` and ``right``.

        :param left: an ``EntityReference`` instance
        :param right: an ``EntityReference`` instance

        :return: a ``float`` value ranged between 0 and 1 which represents the
            probability that ``left`` matches ``right``.
        """
        raise NotImplementedError()
