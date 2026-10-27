from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from typing import NamedTuple, NoReturn, Self

import polars as pl
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._split_generator import (
    MIN_PER_CLASS,
    SplitGenerator,
    _class_counts,
    _pairs_frame,
)

Pair = tuple[RefId, RefId]  # a label-2 pair, stored as (target, bridge)
Row = tuple[RefId, RefId, int]
BridgeSelector = Callable[[Mapping[Pair, int]], AbstractSet[RefId]]
Unit = tuple[tuple[int, ...], int]


class Component(NamedTuple):
    """A connected component of the bridge graph.

    :param clusters: the sorted ids of the clusters in the component
    :param bridges: the sorted bridge records whose footprints form the component
    :param label_two: the number of label-2 pairs of those bridges
    """

    clusters: tuple[int, ...]
    bridges: tuple[RefId, ...]
    label_two: int


def _label_two_right_records(matcher_gt: Mapping[Pair, int]) -> frozenset[RefId]:
    """The default bridge selector: the right record of every label-2 pair."""
    return frozenset(bridge for (_, bridge), label in matcher_gt.items() if label == 2)


def _ref_key(ref: RefId) -> tuple[str, str]:
    return ref.source, str(ref.label)


def _pair_key(pair: Pair) -> tuple[tuple[str, str], tuple[str, str]]:
    return _ref_key(pair[1]), _ref_key(pair[0])


def _find(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[int, int], nodes: list[int]) -> None:
    for node in nodes:
        parent.setdefault(node, node)
    root = _find(parent, nodes[0])
    for node in nodes[1:]:
        parent[_find(parent, node)] = root


def _frame_rows(frame: pl.DataFrame) -> list[Row]:
    return [
        (
            RefId(row["left_id"], row["left_source"]),
            RefId(row["right_id"], row["right_source"]),
            row["label"],
        )
        for row in frame.iter_rows(named=True)
    ]


def _violation(invariant: str, detail: str) -> NoReturn:
    raise RuntimeError(f"invariant {invariant} violated: {detail}")


class BadTripletSplitGenerator(SplitGenerator):
    """Distributes the bad triplets over the splits without leaking bridge text.

    A bridge record ``b`` is recognised by the bridge selector; by default it is
    the right record of a label-2 pair ``(target, b)``. Its footprint is its home
    cluster plus every target cluster: the clusters its text is derived from.
    Every bridge links the clusters of its footprint; the connected components of
    the resulting bridge graph are the units of assignment.

    Components are placed whole, largest first, into the split with the largest
    label-2 deficit against ``split_ratio``. Clusters with label-1 pairs follow,
    placed by their label-1 deficit; the label-1 target of a split is proportional
    to the label-2 pairs it received. The remaining clusters are spread by their
    record count against ``split_ratio``.

    A bridge that cannot keep its footprint in one split is dropped and reported
    in ``orphan_records`` with a reason; dropping is always on.
    """

    def __init__(self, *args, drop_orphans: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not drop_orphans:
            self._log.warning(
                "drop_orphans=False is ignored: orphan bridges are always dropped"
            )
        self.orphan_records: dict[RefId, str] = {}
        self._select_bridges: BridgeSelector = _label_two_right_records

    def with_bridge_selector(self, selector: BridgeSelector) -> Self:
        """Recognise bridge records with *selector* instead of the default."""
        self._select_bridges = selector
        return self

    def generate(self) -> Self:
        self._check_loaded()
        by = self._pairs_by_class()
        if unsupported := by.keys() - {0, 1, 2}:
            raise ValueError(
                f"bad triplet split does not support classes {unsupported}"
            )
        links = self._bridge_links()
        components = self._components(links)
        self._log_components(components)
        split_of = self._assign(components, links)
        self.orphan_records = {}
        self._splits = self._build_splits(by, split_of, by.get(2, []), set())
        self._check_invariants(split_of, links)
        return self

    def _bridge_links(self) -> dict[RefId, list[Pair]]:
        """The validated label-2 pairs of every bridge, both in sorted order."""
        bridges = self._select_bridges(self._matcher_gt)
        links: dict[RefId, list[Pair]] = defaultdict(list)
        label_two = (pair for pair, label in self._matcher_gt.items() if label == 2)
        for pair in sorted(label_two, key=_pair_key):
            self._check_link(pair, bridges)
            links[pair[1]].append(pair)
        self._check_sources(links)
        return dict(links)

    def _check_link(self, pair: Pair, bridges: AbstractSet[RefId]) -> None:
        target, bridge = pair
        if bridge not in bridges:
            raise ValueError(
                f"label-2 pair ({target}, {bridge}): {bridge} is not a bridge"
            )
        if target not in self._cluster_gt or bridge not in self._cluster_gt:
            raise ValueError(f"bridge pair ({target}, {bridge}) lacks cluster labels")
        if self._cluster_gt[target] == self._cluster_gt[bridge]:
            raise ValueError(f"bridge pair ({target}, {bridge}) joins one cluster")

    def _check_sources(self, links: dict[RefId, list[Pair]]) -> None:
        missing = sorted(links.keys() - self._anchored_bridges(links), key=_ref_key)
        if missing:
            raise ValueError(f"bridges without a triplet source pair: {missing}")

    def _anchored_bridges(self, links: dict[RefId, list[Pair]]) -> set[RefId]:
        """The bridges that have at least one triplet source pair."""
        anchored = set()
        for (left, right), label in self._matcher_gt.items():
            if label != 1:
                continue
            ends = ((right, left), (left, right))
            anchored.update(b for b, h in ends if self._is_source(b, h, links))
        return anchored

    def _is_source(self, bridge: RefId, other: RefId, links) -> bool:
        home = self._cluster_gt[bridge] if bridge in links else None
        return home is not None and self._cluster_gt.get(other) == home

    def _footprint(self, bridge: RefId, pairs: list[Pair]) -> list[int]:
        """The sorted clusters the text of *bridge* is derived from."""
        targets = {self._cluster_gt[target] for target, _ in pairs}
        return sorted(targets | {self._cluster_gt[bridge]})

    def _bridge_forest(self, links: dict[RefId, list[Pair]]) -> dict[int, int]:
        """Union-find parents: every bridge joins the clusters of its footprint."""
        parent: dict[int, int] = {}
        for bridge, pairs in links.items():
            _union(parent, self._footprint(bridge, pairs))
        return parent

    def _components(self, links: dict[RefId, list[Pair]]) -> list[Component]:
        """The bridge graph components, sorted by their smallest cluster id."""
        parent = self._bridge_forest(links)
        clusters: dict[int, list[int]] = defaultdict(list)
        for cluster in sorted(parent):
            clusters[_find(parent, cluster)].append(cluster)
        bridges: dict[int, list[RefId]] = defaultdict(list)
        for bridge in links:
            bridges[_find(parent, self._cluster_gt[bridge])].append(bridge)
        components = (
            Component(
                tuple(clusters[root]),
                tuple(members),
                sum(len(links[b]) for b in members),
            )
            for root, members in bridges.items()
        )
        return sorted(components, key=lambda component: component.clusters[0])

    def _ratio_shares(self, total: float) -> dict[str, float]:
        """The fractional share of *total* of every split under ``split_ratio``."""
        weight = sum(self.split_ratio.values())
        return {name: total * r / weight for name, r in self.split_ratio.items()}

    def _log_components(self, components: list[Component]) -> None:
        sizes = [component.label_two for component in components]
        total = sum(sizes)
        self._log.info("bridge graph: %d components", len(components))
        self._log.info(
            "label-2 pairs per component: %s", dict(sorted(Counter(sizes).items()))
        )
        if total:
            self._log.info(
                "largest component share: %.4f (%d/%d)",
                max(sizes) / total,
                max(sizes),
                total,
            )
        targets = self._ratio_shares(total)
        self._log.info(
            "split label-2 targets: %s",
            ", ".join(f"{k}={v:.1f}" for k, v in targets.items()),
        )

    def _assign(
        self, components: list[Component], links: dict[RefId, list[Pair]]
    ) -> dict[int, str]:
        """Assign every cluster to one split; return ``cluster -> split``."""
        split_of: dict[int, str] = {}
        label_two = self._assign_components(components, split_of)
        self._assign_match_clusters(self._match_counts(links), label_two, split_of)
        self._assign_remaining(split_of)
        return split_of

    def _assign_components(
        self, components: list[Component], split_of: dict[int, str]
    ) -> dict[str, float]:
        """Place whole components by label-2 deficit, largest first."""
        ordered = sorted(components, key=lambda c: (-c.label_two, c.clusters[0]))
        targets = self._ratio_shares(sum(c.label_two for c in components))
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        units = [(c.clusters, c.label_two) for c in ordered]
        self._place_greedy(units, targets, assigned, split_of)
        self._log.info(
            "split label-2 assigned: %s",
            ", ".join(f"{k}={assigned[k]:.0f}/{v:.1f}" for k, v in targets.items()),
        )
        return assigned

    def _match_counts(self, links: dict[RefId, list[Pair]]) -> Counter[int]:
        """Within-cluster label-1 pairs that contain no bridge, per cluster."""
        counts: Counter[int] = Counter()
        for (left, right), label in self._matcher_gt.items():
            if label != 1 or left in links or right in links:
                continue
            if self._cluster_gt.get(left) == self._cluster_gt.get(right):
                counts[self._cluster_gt.get(left)] += 1
        return counts

    def _assign_match_clusters(
        self,
        counts: Counter[int],
        label_two: dict[str, float],
        split_of: dict[int, str],
    ) -> None:
        """Place clusters with label-1 pairs by label-1 deficit, largest first."""
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        for cluster, split in split_of.items():
            assigned[split] += counts[cluster]
        free = sorted(
            (c for c in counts if c not in split_of), key=lambda c: (-counts[c], c)
        )
        targets = self._match_targets(counts.total(), label_two)
        self._place_greedy(
            [((c,), counts[c]) for c in free], targets, assigned, split_of
        )

    def _match_targets(
        self, total: int, label_two: dict[str, float]
    ) -> dict[str, float]:
        """Label-1 targets proportional to the label-2 pairs of every split."""
        received = sum(label_two.values())
        if not received:
            return self._ratio_shares(total)
        return {name: total * n / received for name, n in label_two.items()}

    def _assign_remaining(self, split_of: dict[int, str]) -> None:
        """Spread the clusters without pairs by record count against the ratio."""
        rest = [c for c in self._clusters if c not in split_of]
        sizes = {c: len(self._clusters[c]) for c in rest}
        units = [((c,), sizes[c]) for c in sorted(rest, key=lambda c: (-sizes[c], c))]
        targets = self._ratio_shares(sum(sizes.values()))
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        self._place_greedy(units, targets, assigned, split_of)

    def _place_greedy(
        self,
        units: list[Unit],
        targets: dict[str, float],
        assigned: dict[str, float],
        split_of: dict[int, str],
    ) -> None:
        """Give every unit to the split with the largest deficit."""
        for clusters, weight in units:
            split = self._most_deficient(targets, assigned)
            assigned[split] += weight
            split_of.update(dict.fromkeys(clusters, split))

    def _most_deficient(
        self, targets: dict[str, float], assigned: dict[str, float]
    ) -> str:
        """The split furthest below its target (``self._rng`` breaks ties)."""
        deficits = {name: targets[name] - assigned[name] for name in self.split_ratio}
        best = max(deficits.values())
        candidates = [name for name, deficit in deficits.items() if deficit == best]
        return candidates[self._rng.randint(len(candidates))]

    def _check_invariants(
        self, split_of: dict[int, str], links: dict[RefId, list[Pair]]
    ) -> None:
        """Raise on a violated invariant; warn on scarce label-2 pairs."""
        rows = {name: _frame_rows(frame) for name, frame in self._splits.items()}
        for name, split_rows in rows.items():
            self._check_disjoint(name, split_rows, split_of)
            self._check_footprints(name, split_rows, split_of, links)
        self._check_label_two(rows, split_of, links)
        self._warn_scarce_label_two(rows)

    def _check_disjoint(
        self, name: str, rows: list[Row], split_of: dict[int, str]
    ) -> None:
        """I4: no row spans two splits; I1: every row lies in its clusters' split."""
        for left, right, _ in rows:
            splits = {split_of[self._cluster_gt[ref]] for ref in (left, right)}
            if len(splits) > 1:
                _violation("I4", f"row ({left}, {right}) spans {sorted(splits)}")
            if splits != {name}:
                _violation("I1", f"row ({left}, {right}) in {name} belongs to {splits}")

    def _check_footprints(
        self,
        name: str,
        rows: list[Row],
        split_of: dict[int, str],
        links: dict[RefId, list[Pair]],
    ) -> None:
        """I3: orphans are mentioned nowhere; I2: footprints stay in the split."""
        mentioned = {ref for left, right, _ in rows for ref in (left, right)}
        bridges = sorted(mentioned & links.keys(), key=_ref_key)
        if orphans := [b for b in bridges if b in self.orphan_records]:
            _violation("I3", f"orphan bridges {orphans} appear in {name}")
        for bridge in bridges:
            footprint = self._footprint(bridge, links[bridge])
            if outside := [c for c in footprint if split_of[c] != name]:
                _violation("I2", f"{bridge} in {name}: clusters {outside} elsewhere")

    def _check_label_two(
        self,
        rows: dict[str, list[Row]],
        split_of: dict[int, str],
        links: dict[RefId, list[Pair]],
    ) -> None:
        """I5: every label-2 pair of a kept bridge appears in the bridge's split."""
        present = {name: set(split_rows) for name, split_rows in rows.items()}
        for bridge, pairs in links.items():
            if bridge in self.orphan_records:
                continue
            name = split_of[self._cluster_gt[bridge]]
            if missing := [p for p in pairs if (*p, 2) not in present[name]]:
                _violation("I5", f"{bridge} in {name}: label-2 pairs {missing} missing")

    def _warn_scarce_label_two(self, rows: dict[str, list[Row]]) -> None:
        """I8: warn about splits with fewer than ``MIN_PER_CLASS`` label-2 pairs."""
        for name, split_rows in rows.items():
            count = sum(1 for *_, label in split_rows if label == 2)
            if count < MIN_PER_CLASS:
                self._log.warning(
                    "split %s has %d label-2 pairs, fewer than MIN_PER_CLASS=%d",
                    name,
                    count,
                    MIN_PER_CLASS,
                )

    def _cross_cluster_pairs(self, records: list[RefId]) -> int:
        sizes = Counter(self._cluster_gt[r] for r in records)
        n = len(records)
        return (n * n - sum(s * s for s in sizes.values())) // 2

    def _build_splits(
        self,
        by: dict[int, list[Pair]],
        split_of: dict[int, str],
        label_two: list[Pair],
        excluded: set[RefId],
    ) -> dict:
        """Assemble the frames; generated label-0 rows pad the kept positive rows."""
        frames = {}
        empty = []
        for name in self.split_ratio:
            records = [
                ref
                for ref, cluster in self._cluster_gt.items()
                if split_of[cluster] == name and ref not in excluded
            ]
            members = set(records)
            n2 = [
                pair for pair in label_two if pair[0] in members and pair[1] in members
            ]
            n1 = [
                pair
                for pair in by.get(1, [])
                if pair[0] in members and pair[1] in members
            ]
            if n2 and self.match_bridge_ratio > 0:
                n1 = [
                    tuple(pair)
                    for pair in self._select_diverse(
                        n1, round(self.match_bridge_ratio * len(n2))
                    )
                ]
            if not n1 and not n2:
                raise RuntimeError(f"split {name} has no positive pairs")
            n0 = min(
                round(self.neg_pos_ratio * (len(n1) + len(n2))),
                self._cross_cluster_pairs(records),
            )
            negatives = (
                self._generate_negatives(n0, set(self._matcher_gt), records)
                if n0 > 0
                else []
            )
            frames[name] = _pairs_frame({0: negatives, 1: n1, 2: n2})
            if not n2:
                empty.append(name)
            self._log.info(
                "split %s: %d records, %s",
                name,
                len(records),
                _class_counts(frames[name]),
            )
        if empty:
            self._log.warning(
                "splits without bridges (label 2 not evaluable): %s", empty
            )
        return frames
