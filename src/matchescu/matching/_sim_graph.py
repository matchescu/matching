import itertools
from enum import StrEnum
from functools import partial
from typing import Generator

import networkx as nx

from matchescu.matching.similarity import Similarity
from matchescu.typing import EntityReference


class EdgeType(StrEnum):
    MATCH = "match"
    POTENTIAL_MATCH = "potential_match"


class SimilarityGraph:
    def __init__(
        self, similarity: Similarity, max_non_match: float, min_match: float
    ) -> None:
        self.__g = nx.DiGraph()
        self.__sim = similarity
        self.__max_bad = max_non_match
        self.__min_good = min_match
        self.__match_count = 0
        self.__potential_match_count = 0
        self.__non_match_count = 0

    def __repr__(self):
        return "SimilarityGraph(nodes={}, edges={}, match={}, non_match={}, maybe={})".format(
            len(self.__g.nodes),
            len(self.__g.edges),
            self.__match_count,
            self.__non_match_count,
            self.__potential_match_count,
        )

    @property
    def nodes(self):
        """Returns the nodes of the graph."""
        return self.__g.nodes

    @property
    def edges(self):
        """Returns the edges of the graph along with their similarity weights and types."""
        return self.__g.edges(data=True)

    @property
    def match_count(self):
        return self.__match_count

    @property
    def potential_match_count(self):
        return self.__potential_match_count

    @property
    def non_match_count(self):
        return self.__non_match_count

    def add(self, left: EntityReference, right: EntityReference) -> "SimilarityGraph":
        """Adds an edge between two entities based on similarity thresholds."""
        if left not in self.__g:
            self.__g.add_node(left)
        if right not in self.__g:
            self.__g.add_node(right)

        sim_score = self.__sim(left, right)
        if sim_score >= self.__min_good:
            self.__g.add_edge(left, right, weight=sim_score, type=EdgeType.MATCH)
            self.__match_count += 1
        elif self.__max_bad <= sim_score < self.__min_good:
            self.__g.add_edge(
                left, right, weight=sim_score, type=EdgeType.POTENTIAL_MATCH
            )
            self.__potential_match_count += 1
        else:
            self.__non_match_count += 1
        return self

    @staticmethod
    def __has_expected_type(edge: tuple, edge_type: EdgeType) -> bool:
        _, __, data = edge
        return data.get("type") == edge_type

    def edges_by_type(
        self, edge_type: EdgeType
    ) -> Generator[tuple[EntityReference, EntityReference], None, None]:
        yield from itertools.starmap(
            lambda a, b, _: (a, b),
            filter(partial(self.__has_expected_type, edge_type=edge_type), self.edges),
        )

    def matches(self) -> Generator[tuple[EntityReference, EntityReference], None, None]:
        yield from self.edges_by_type(EdgeType.MATCH)

    def potential_matches(
        self,
    ) -> Generator[tuple[EntityReference, EntityReference], None, None]:
        yield from self.edges_by_type(EdgeType.POTENTIAL_MATCH)

    def is_match(self, left: EntityReference, right: EntityReference) -> bool:
        data = self.__g.get_edge_data(left, right, default={})
        return data.get("type") == EdgeType.MATCH

    def is_potential_match(self, left: EntityReference, right: EntityReference) -> bool:
        data = self.__g.get_edge_data(left, right, default={})
        return data.get("type") == EdgeType.POTENTIAL_MATCH

    def is_non_match(self, left: EntityReference, right: EntityReference) -> bool:
        return (left, right) not in self.__g.edges
