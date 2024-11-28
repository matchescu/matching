import pytest

from matchescu.matching.similarity import Similarity


class SimilarityStub(Similarity):
    _DEFAULT_SCORE = 0.5

    def __init__(self, sim_score: float = _DEFAULT_SCORE):
        self._sim_score = sim_score

    def _compute_similarity(self, _, __) -> float:
        return self._sim_score


@pytest.fixture
def similarity_stub(request) -> Similarity:
    is_parameterized_with_sim_score = (
        hasattr(request, "param")
        and request.param is not None
        and isinstance(request.param, (int, float))
    )
    return SimilarityStub(
        sim_score=(
            request.param
            if is_parameterized_with_sim_score
            else SimilarityStub._DEFAULT_SCORE
        )
    )
