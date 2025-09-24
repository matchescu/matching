from typing import Any

from matchescu.matching.similarity._common import Similarity


class ExactMatch(Similarity[int]):
    def __init__(self):
        super().__init__(-1, 0)

    def _compute_similarity(self, a: Any, b: Any) -> int:
        return 1 if a == b else 0
