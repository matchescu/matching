import copy
from dataclasses import dataclass, field
from typing import Set

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId, EntityReference as Ref


@dataclass(eq=True, unsafe_hash=True)
class Split:
    comparison_space: BinaryComparisonSpace
    matcher_labels: dict[tuple[RefId, RefId], int]
    gt_clusters: dict[int, Set[RefId]]
    id_cluster_map: dict[RefId, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.id_cluster_map = {
            ref_id: cluster_id
            for cluster_id, ref_ids in self.gt_clusters.items()
            for ref_id in ref_ids
        }

    def to_partition(self) -> frozenset[frozenset[RefId]]:
        """Format the ``*gt_clusters*`` as an algebraic partition."""
        return frozenset(
            frozenset(x for x in cluster) for cluster in self.gt_clusters.values()
        )

    def to_comparison_labels(
        self, id_table: IdTable
    ) -> tuple[list[tuple[Ref, Ref]], list[int]]:
        """Provide two lists based on ``self.comparison_space``.

        The first list contains pairs of compared references
        (``*comparison_space*`` contains IDs).
        The second list contains the corresponding comparison labels: the label
        assigned to the comparison in ``*matcher_labels*`` or zero otherwise.
        The returned list is suitable for training supervised matchers.

        :param id_table: retrieve entity references by their ID.
        """
        pairs: list[tuple[Ref, Ref]] = []
        labels: list[int] = []
        for ref_ids in self.comparison_space:
            left, right = id_table.get_all(ref_ids)
            label = self.matcher_labels.get(ref_ids, 0)
            pairs.append((left, right))
            labels.append(label)
        return pairs, labels

    @classmethod
    def merge(cls, splits: list["Split"]) -> "Split":
        if splits is None or len(splits) == 0:
            raise ValueError()
        accumulator = copy.deepcopy(splits[0])
        for split in splits[1:]:
            for x, y in split.comparison_space:
                accumulator.comparison_space.put(x, y)
            accumulator.matcher_labels.update(split.matcher_labels)
            accumulator.gt_clusters.update(split.gt_clusters)
            accumulator.id_cluster_map.update(split.id_cluster_map)
        return accumulator
