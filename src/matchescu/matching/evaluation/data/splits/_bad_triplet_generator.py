from collections import Counter, defaultdict
from typing import NamedTuple

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._split_generator import SplitGenerator, _class_counts, _pairs_frame

Pair = tuple[RefId, RefId]  # a label-2 pair, stored as (target, bridge)


class BadTriplet(NamedTuple):
    """A possible bad triplet

    :param bridge_ref_id: the reference id of the bridge record itself
    :param source: the id of the cluster containing the bridge record
    :param target_links: a ``dict`` containing a mapping of cluster IDs to links
        formed from those clusters to the bridge record.
    """

    bridge_ref_id: RefId  # ambiguous record; it actually belongs to ``home``
    source: int
    target_links: dict[int, list[Pair]]  # distinct target clusters -> (target, record) pairs


class Component(NamedTuple):
    """A connected component of the bridge graph.

    :param clusters: the sorted ids of the clusters in the component
    :param bridges: the sorted bridge records whose footprints form the component
    :param label_two: the number of label-2 pairs of those bridges
    """

    clusters: tuple[int, ...]
    bridges: tuple[RefId, ...]
    label_two: int


def _ref_key(ref: RefId) -> tuple[str, str]:
    return ref.source, str(ref.label)


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


class BadTripletSplitGenerator(SplitGenerator):
    """Distributes the possible bad triplets over the splits.

    Every ``(source, bridge) = 1`` and ``(target, bridge) = 2`` link of a bridge
    record ``b`` delimits a bad triplet ``(source, target, b)``. The distributed
    units are the unique combinations of ``b``'s home cluster (the one ``b``
    belongs to) with one of its target clusters (a cluster ``b`` resembles but
    does not belong to). All links of a bridge record are kept: several source and
    several target links are legal, and two clusters may be bridged by several
    records.

    A bridge record is realized toward exactly one target cluster: both clusters
    land whole in a single split, chosen by the ``split_ratio`` quotas. Any other
    target cluster of the record may not enter that split (rule 4), so ambiguity
    material cannot spill from one split into another; such a cluster keeps its
    remaining material wherever it is placed, and later bridge records may still
    select it. When the split ratio cannot satisfy rule 4, the target clusters are
    placed together and all their label-2 pairs are kept: no label-2 pair is ever
    dropped from a split.

    A bridge record whose home cluster cannot share a split with any of its target
    clusters is reported in ``orphan_records``; its label-2 rows are cross-split by
    construction. ``drop_orphans`` removes such records from the frames, except as
    target records of label-2 rows that do appear.
    """

    def __init__(self, *args, drop_orphans: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_orphans = drop_orphans
        self.orphan_records: set[RefId] = set()

    def generate(self) -> "BadTripletSplitGenerator":
        self._check_loaded()
        by = self._pairs_by_class()
        if unsupported := by.keys() - {0, 1, 2}:
            raise ValueError(
                f"bad triplet split does not support classes {unsupported}"
            )
        self._log_components(self._components())
        split_of = self._distribute(self._triplets())
        label_two = [
            pair
            for pair in by.get(2, [])
            if split_of[self._cluster_gt[pair[0]]]
            == split_of[self._cluster_gt[pair[1]]]
        ]
        excluded = (
            self.orphan_records - {pair[0] for pair in label_two}
            if self.drop_orphans
            else set()
        )
        self._splits = self._build_splits(by, split_of, label_two, excluded)
        return self

    def _label_two_by_bridge(self) -> dict[RefId, list[Pair]]:
        """The label-2 pairs of every bridge, both in sorted order."""
        result: dict[RefId, list[Pair]] = defaultdict(list)
        pairs = (pair for pair, label in self._matcher_gt.items() if label == 2)
        for pair in sorted(pairs, key=lambda p: (_ref_key(p[1]), _ref_key(p[0]))):
            result[pair[1]].append(pair)
        return result

    def _footprint(self, bridge: RefId, pairs: list[Pair]) -> list[int]:
        """The sorted clusters the text of *bridge* is derived from."""
        targets = {self._cluster_gt[target] for target, _ in pairs}
        return sorted(targets | {self._cluster_gt[bridge]})

    def _bridge_forest(self, by_bridge: dict[RefId, list[Pair]]) -> dict[int, int]:
        """Union-find parents: every bridge joins the clusters of its footprint."""
        parent: dict[int, int] = {}
        for bridge, pairs in by_bridge.items():
            _union(parent, self._footprint(bridge, pairs))
        return parent

    def _components(self) -> list[Component]:
        """The bridge graph components, sorted by their smallest cluster id."""
        by_bridge = self._label_two_by_bridge()
        parent = self._bridge_forest(by_bridge)
        clusters: dict[int, list[int]] = defaultdict(list)
        for cluster in sorted(parent):
            clusters[_find(parent, cluster)].append(cluster)
        bridges: dict[int, list[RefId]] = defaultdict(list)
        for bridge in by_bridge:
            bridges[_find(parent, self._cluster_gt[bridge])].append(bridge)
        components = (
            Component(
                tuple(clusters[root]),
                tuple(members),
                sum(len(by_bridge[b]) for b in members),
            )
            for root, members in bridges.items()
        )
        return sorted(components, key=lambda component: component.clusters[0])

    def _label_two_targets(self, total: int) -> dict[str, float]:
        """The fractional label-2 share of every split under ``split_ratio``."""
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
        targets = self._label_two_targets(total)
        self._log.info(
            "split label-2 targets: %s",
            ", ".join(f"{k}={v:.1f}" for k, v in targets.items()),
        )

    def _triplets(self) -> dict[RefId, BadTriplet]:
        """Group every available ``(source, bridge, target)`` triplet per bridge record."""
        label_one = {
            ref
            for pair, label in self._matcher_gt.items()
            if label == 1
            for ref in pair
        }
        targets: dict[RefId, dict[int, list[Pair]]] = {}
        for (target, bridge), label in self._matcher_gt.items():
            if label != 2:
                continue
            home = self._cluster_gt.get(bridge)
            cluster = self._cluster_gt.get(target)
            if home is None or cluster is None:
                raise ValueError(
                    f"bridge pair ({target}, {bridge}) lacks cluster labels"
                )
            if home == cluster:
                raise ValueError(f"bridge pair ({target}, {bridge}) joins one cluster")
            targets.setdefault(bridge, {}).setdefault(cluster, []).append(
                (target, bridge)
            )
        triplets = {}
        for record, by_target in targets.items():
            if record not in label_one:
                self._log.warning(
                    "bridge %s has no (source, bridge) label-1 link", record
                )
                continue
            triplets[record] = BadTriplet(record, self._cluster_gt[record], by_target)
        return triplets

    def _distribute(self, bridges: dict[RefId, BadTriplet]) -> dict[int, str]:
        """Place one triplet unit per bridge record; return ``cluster -> split``."""
        names = list(self.split_ratio)
        self.orphan_records = set()
        split_of: dict[int, str] = {}
        forbidden: dict[int, set[str]] = defaultdict(set)
        remaining = dict(zip(names, self._quota_counts(len(bridges), eval_first=False)))
        placed = list(bridges.values())
        for index in self._rng.permutation(len(placed)):
            bridge = placed[int(index)]
            placement = self._place(bridge, split_of, forbidden, remaining, names)
            if placement is None:
                self.orphan_records.add(bridge.bridge_ref_id)
                continue
            target, split = placement
            fresh = bridge.source not in split_of and target not in split_of
            split_of[bridge.source] = split_of[target] = split
            if fresh:
                remaining[split] -= 1
            for loser in bridge.target_links:
                if loser != target and loser not in split_of:
                    forbidden[loser].add(split)
        free = [c for c in self._clusters if c not in split_of]
        free = [free[int(i)] for i in self._rng.permutation(len(free))]
        free_remaining = dict(zip(names, self._quota_counts(len(free))))
        for cluster in free:
            allowed = [name for name in names if name not in forbidden[cluster]]
            if not allowed:
                self._log.warning(
                    "cluster %s would break rule 4 in every split", cluster
                )
                allowed = names
            split = self._pick_split(allowed, free_remaining)
            free_remaining[split] -= 1
            split_of[cluster] = split
        return split_of

    def _place(
        self,
        bridge: BadTriplet,
        split_of: dict[int, str],
        forbidden: dict[int, set[str]],
        remaining: dict[str, int],
        names: list[str],
    ) -> tuple[RefId, str] | None:
        """One target cluster of *bridge* and the split both clusters join."""
        targets = list(bridge.target_links)
        order = [targets[int(i)] for i in self._rng.permutation(len(targets))]
        for strict in (True, False):
            for target in order:
                split = self._split_for(
                    bridge, target, strict, split_of, forbidden, remaining, names
                )
                if split is None:
                    continue
                if not strict:
                    self._log.warning(
                        "bridge %s: rule 4 cannot be satisfied for split %s",
                        bridge.bridge_ref_id,
                        split,
                    )
                return target, split
        self._log.warning(
            "bridge %s cannot be placed with any target cluster", bridge.bridge_ref_id
        )
        return None

    def _split_for(
        self,
        bridge: BadTriplet,
        target: int,
        strict: bool,
        split_of: dict[int, str],
        forbidden: dict[int, set[str]],
        remaining: dict[str, int],
        names: list[str],
    ) -> str | None:
        """The split *bridge* and *target* would share, or None when unavailable."""
        home = bridge.source
        if (
            home in split_of
            and target in split_of
            and split_of[home] != split_of[target]
        ):
            return None
        if home in split_of:
            split = split_of[home]
        elif target in split_of:
            split = split_of[target]
        elif strict:
            blocked = {
                split_of[o] for o in bridge.target_links if o != target and o in split_of
            }
            allowed = [
                name
                for name in names
                if name not in forbidden[home]
                and name not in forbidden[target]
                and name not in blocked
            ]
            return self._pick_split(allowed, remaining) if allowed else None
        else:
            return self._pick_split(names, remaining)
        blocked = any(
            other != target and split_of.get(other) == split for other in bridge.target_links
        )
        if strict and (
            blocked or split in forbidden[home] or split in forbidden[target]
        ):
            return None
        return split

    def _pick_split(self, allowed: list[str], remaining: dict[str, int]) -> str:
        """The allowed split with the most remaining quota (rng tie-break)."""
        best = max(remaining[name] for name in allowed)
        candidates = [name for name in allowed if remaining[name] == best]
        return candidates[self._rng.randint(len(candidates))]

    def _quota_counts(self, items: int, eval_first: bool = True) -> list[int]:
        """Per-split item quota from ``split_ratio``."""
        ratios = list(self.split_ratio.values())
        if items < len(ratios):
            if not eval_first:
                return [items] + [0] * (len(ratios) - 1)
            return [0] * (len(ratios) - items) + [1] * items
        return self._split_counts(items, ratios)

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
