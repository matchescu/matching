from unittest.mock import MagicMock

import pytest

from matchescu.matching import SimilarityGraph, Similarity
from matchescu.matching._sim_graph import EdgeType


@pytest.fixture(scope="module")
def comparison_space(ref):
    return {(ref(1, "a"), ref(1, "b")), (ref(1, "b"), ref(1, "a"))}


@pytest.fixture
def mock_similarity(request):
    mock = MagicMock(name="mock_similarity", spec=Similarity)
    mock.return_value = request.param if hasattr(request, "param") else 0.5
    return mock


@pytest.fixture
def sim_graph(mock_similarity):
    return SimilarityGraph(mock_similarity, 0.25, 0.75)


def test_sim_graph_calls_sim(comparison_space, sim_graph, mock_similarity):
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert mock_similarity.call_count == len(comparison_space)


def test_add_match_edge(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.75
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert sim_graph.match_count == len(comparison_space)
    assert sim_graph.potential_match_count == 0
    assert sim_graph.non_match_count == 0
    assert len(sim_graph.edges) == 2


def test_add_potential_match_edge(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.5
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert sim_graph.potential_match_count == len(comparison_space)
    assert sim_graph.match_count == 0
    assert sim_graph.non_match_count == 0
    assert len(sim_graph.edges) == 2


def test_skip_non_match(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.24
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert sim_graph.match_count == 0
    assert sim_graph.potential_match_count == 0
    assert sim_graph.non_match_count == len(comparison_space)
    assert len(sim_graph.edges) == 0


@pytest.mark.parametrize(
    "mock_similarity,edge_type",
    [(0.5, EdgeType.POTENTIAL_MATCH), (0.75, EdgeType.MATCH)],
    indirect=["mock_similarity"],
)
def test_filter_by_edge_type(comparison_space, sim_graph, mock_similarity, edge_type):
    for a, b in comparison_space:
        sim_graph.add(a, b)

    actual = list(sim_graph.edges_by_type(edge_type))

    assert actual == list(comparison_space)


def test_is_match(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.75
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert all(sim_graph.is_match(a, b) for a, b in comparison_space)
    assert all(not sim_graph.is_non_match(a, b) for a, b in comparison_space)


def test_is_potential_match(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.74
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert all(sim_graph.is_potential_match(a, b) for a, b in comparison_space)
    assert all(not sim_graph.is_non_match(a, b) for a, b in comparison_space)


def test_is_non_match(comparison_space, sim_graph, mock_similarity):
    mock_similarity.return_value = 0.24
    for a, b in comparison_space:
        sim_graph.add(a, b)

    assert all(sim_graph.is_non_match(a, b) for a, b in comparison_space)
