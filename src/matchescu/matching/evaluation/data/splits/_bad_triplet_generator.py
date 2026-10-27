import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import NamedTuple, NoReturn, Self

import polars as pl
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._split_generator import MIN_PER_CLASS, SplitGenerator, _pairs_frame

Pair = tuple[RefId, RefId]
Row = tuple[RefId, RefId, int]
BridgeSelector = Callable[[Mapping[Pair, int]], AbstractSet[RefId]]
Links = dict[RefId, list[Pair]]
Unit = tuple[tuple[int, ...], int]

NEGATIVE = "negative"
TRIPLET_SOURCE = "triplet_source"
EXTRA_MATCH = "extra_match"
TRIPLET_TARGET = "triplet_target"
CATEGORY_LABELS = {NEGATIVE: 0, TRIPLET_SOURCE: 1, EXTRA_MATCH: 1, TRIPLET_TARGET: 2}


class Component(NamedTuple):
    """A connected component of the bridge graph.

    :param clusters: the sorted ids of the clusters in the component
    :param bridges: the sorted bridge records whose footprints form the component
    :param label_two: the number of label-2 pairs of those bridges
    """

    clusters: tuple[int, ...]
    bridges: tuple[RefId, ...]
    label_two: int


class CategorizedPair(NamedTuple):
    """A split row together with the role it plays in a bad triplet.

    :param pair: the compared records, oriented as in the matcher ground truth
    :param category: ``negative``, ``triplet_source``, ``extra_match`` or
        ``triplet_target``
    :param bridge: the bridge record of a triplet row, ``None`` for other rows
    """

    pair: Pair
    category: str
    bridge: RefId | None = None


def _label_two_right_records(matcher_gt: Mapping[Pair, int]) -> frozenset[RefId]:
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


def _labelled_frame(rows: list[CategorizedPair]) -> pl.DataFrame:
    samples: dict[int, list[Pair]] = defaultdict(list)
    for row in rows:
        samples[CATEGORY_LABELS[row.category]].append(row.pair)
    return _pairs_frame(samples)


def _meta_record(row: CategorizedPair) -> dict:
    left, right = row.pair
    return {
        "left_id": left.label,
        "left_source": left.source,
        "right_id": right.label,
        "right_source": right.source,
        "category": row.category,
        "bridge_id": row.bridge.label if row.bridge else None,
        "bridge_source": row.bridge.source if row.bridge else None,
    }


def _violation(detail: str) -> NoReturn:
    raise RuntimeError(f"bad triplet split is inconsistent: {detail}")


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
    record count against ``split_ratio``. Every split must receive label-2 pairs.

    Every bridge keeps all its label-2 pairs (triplet targets) and a number of
    label-1 pairs with members of its home cluster (triplet sources), drawn at
    random. Label-1 pairs without a bridge (extra matches) fill the remaining
    label-1 budget of the split, ``match_bridge_ratio`` times its label-2 pairs.

    A bridge that cannot keep its footprint in one split is dropped and reported
    in ``orphan_records`` with a reason; dropping is always on.

    :param drop_orphans: kept for compatibility; ``False`` is ignored with a warning
    :param source_links_per_bridge: the most triplet source pairs a bridge keeps;
        ``None`` keeps as many as ``match_bridge_ratio`` and the home cluster allow
    """

    def __init__(
        self,
        *args,
        drop_orphans: bool = True,
        source_links_per_bridge: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not drop_orphans:
            self._log.warning(
                "drop_orphans=False is ignored: orphan bridges are always dropped"
            )
        self.source_links_per_bridge = source_links_per_bridge
        self.orphan_records: dict[RefId, str] = {}
        self._select_bridges: BridgeSelector = _label_two_right_records
        self._categorized: dict[str, list[CategorizedPair]] = {}

    def with_bridge_selector(self, selector: BridgeSelector) -> Self:
        """Recognise bridge records with *selector* instead of the default."""
        self._select_bridges = selector
        return self

    def generate(self) -> Self:
        """Assign the clusters to splits and build the labelled rows of each split."""
        self._check_loaded()
        by = self._pairs_by_class()
        if unsupported := by.keys() - {0, 1, 2}:
            raise ValueError(
                f"bad triplet split does not support classes {unsupported}"
            )
        links = self._bridge_links()
        sources = self._source_pairs(links)
        self._check_sources(links, sources)
        components = self._components(links)
        self._log_components(components)
        split_of = self._assign(components, links)
        self.orphan_records = {}
        self._splits = self._build_splits(by.get(1, []), split_of, links, sources)
        self._check_invariants(split_of, links, sources)
        return self

    def save(
        self,
        output_dir: str | os.PathLike,
        prefix: str = "",
        file_names: Mapping[str, str] | None = None,
    ) -> Self:
        """Write the split CSV files, then one ``{prefix}_{split}_meta.csv`` per split.

        A meta file lists the rows of its split in the same order, with the
        bad-triplet category of every row and the bridge of every triplet row.
        """
        super().save(output_dir, prefix, file_names)
        for name, rows in self._categorized.items():
            path = Path(output_dir) / self._file_name(f"{name}_meta", prefix, None)
            records = [_meta_record(row) for row in rows]
            pl.DataFrame(records, infer_schema_length=None).write_csv(path)
        return self

    def _bridge_links(self) -> Links:
        bridges = self._select_bridges(self._matcher_gt)
        links: Links = defaultdict(list)
        label_two = (pair for pair, label in self._matcher_gt.items() if label == 2)
        for pair in sorted(label_two, key=_pair_key):
            self._check_link(pair, bridges)
            links[pair[1]].append(pair)
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

    def _source_pairs(self, links: Links) -> Links:
        sources: Links = defaultdict(list)
        label_one = (pair for pair, label in self._matcher_gt.items() if label == 1)
        for pair in sorted(label_one, key=_pair_key):
            for bridge in self._bridges_anchored_by(pair, links):
                sources[bridge].append(pair)
        return sources

    def _bridges_anchored_by(self, pair: Pair, links: Links) -> list[RefId]:
        left, right = pair
        if self._cluster_gt.get(left) != self._cluster_gt.get(right):
            return []
        return [ref for ref in (right, left) if ref in links]

    @staticmethod
    def _check_sources(links: Links, sources: Links) -> None:
        missing = sorted(links.keys() - sources.keys(), key=_ref_key)
        if missing:
            raise ValueError(f"bridges without a triplet source pair: {missing}")

    def _source_quota(self, bridge: RefId, links: Links) -> int:
        home_size = len(self._clusters[self._cluster_gt[bridge]])
        quota = min(home_size - 1, round(self.match_bridge_ratio * len(links[bridge])))
        if self.source_links_per_bridge is None:
            return quota
        return min(quota, self.source_links_per_bridge)

    def _footprint(self, bridge: RefId, pairs: list[Pair]) -> list[int]:
        targets = {self._cluster_gt[target] for target, _ in pairs}
        return sorted(targets | {self._cluster_gt[bridge]})

    def _bridge_forest(self, links: Links) -> dict[int, int]:
        parent: dict[int, int] = {}
        for bridge, pairs in links.items():
            _union(parent, self._footprint(bridge, pairs))
        return parent

    def _components(self, links: Links) -> list[Component]:
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

    def _assign(self, components: list[Component], links: Links) -> dict[int, str]:
        split_of: dict[int, str] = {}
        label_two = self._assign_components(components, split_of)
        self._assign_match_clusters(self._match_counts(links), label_two, split_of)
        self._assign_remaining(split_of)
        return split_of

    def _assign_components(
        self, components: list[Component], split_of: dict[int, str]
    ) -> dict[str, float]:
        ordered = sorted(components, key=lambda c: (-c.label_two, c.clusters[0]))
        targets = self._ratio_shares(sum(c.label_two for c in components))
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        units = [(c.clusters, c.label_two) for c in ordered]
        self._place_greedy(units, targets, assigned, split_of)
        self._log.info(
            "split label-2 assigned: %s",
            ", ".join(f"{k}={assigned[k]:.0f}/{v:.1f}" for k, v in targets.items()),
        )
        if empty := [name for name, count in assigned.items() if not count]:
            raise ValueError(f"splits without label-2 pairs: {empty}")
        return assigned

    def _match_counts(self, links: Links) -> Counter[int]:
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

    @staticmethod
    def _match_targets(total: int, label_two: dict[str, float]) -> dict[str, float]:
        received = sum(label_two.values())
        return {name: total * n / received for name, n in label_two.items()}

    def _assign_remaining(self, split_of: dict[int, str]) -> None:
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
        for clusters, weight in units:
            split = self._most_deficient(targets, assigned)
            assigned[split] += weight
            split_of.update(dict.fromkeys(clusters, split))

    def _most_deficient(
        self, targets: dict[str, float], assigned: dict[str, float]
    ) -> str:
        deficits = {name: targets[name] - assigned[name] for name in self.split_ratio}
        best = max(deficits.values())
        candidates = [name for name, deficit in deficits.items() if deficit == best]
        return candidates[self._rng.randint(len(candidates))]

    def _build_splits(
        self,
        label_one: list[Pair],
        split_of: dict[int, str],
        links: Links,
        sources: Links,
    ) -> dict[str, pl.DataFrame]:
        self._categorized = {
            name: self._categorize(name, label_one, split_of, links, sources)
            for name in self.split_ratio
        }
        return {name: _labelled_frame(rows) for name, rows in self._categorized.items()}

    def _categorize(
        self,
        name: str,
        label_one: list[Pair],
        split_of: dict[int, str],
        links: Links,
        sources: Links,
    ) -> list[CategorizedPair]:
        records = [ref for ref, c in self._cluster_gt.items() if split_of[c] == name]
        positives = self._positives(name, set(records), label_one, links, sources)
        rows = self._negatives(records, len(positives)) + positives
        categories = Counter(row.category for row in rows)
        self._log.info(
            "split %s: %d records, %s",
            name,
            len(records),
            dict(sorted(categories.items())),
        )
        return rows

    def _positives(
        self,
        name: str,
        members: set[RefId],
        label_one: list[Pair],
        links: Links,
        sources: Links,
    ) -> list[CategorizedPair]:
        bridges = [bridge for bridge in links if bridge in members]
        targets = [
            CategorizedPair(p, TRIPLET_TARGET, b) for b in bridges for p in links[b]
        ]
        anchored = [
            CategorizedPair(p, TRIPLET_SOURCE, b)
            for b in bridges
            for p in self._anchor(b, sources[b], links)
        ]
        budget = round(self.match_bridge_ratio * len(targets)) - len(anchored)
        extras = self._extra_matches(name, members, label_one, links, budget)
        return anchored + extras + targets

    def _anchor(
        self, bridge: RefId, candidates: list[Pair], links: Links
    ) -> list[Pair]:
        quota = self._source_quota(bridge, links)
        picks = sorted(self._rng.choice(len(candidates), size=quota, replace=False))
        return [candidates[i] for i in picks]

    def _extra_matches(
        self,
        name: str,
        members: set[RefId],
        label_one: list[Pair],
        links: Links,
        budget: int,
    ) -> list[CategorizedPair]:
        if budget < 0:
            self._log.warning(
                "split %s: triplet source pairs exceed the label-1 budget by %d; "
                "no extra matches added",
                name,
                -budget,
            )
            return []
        pool = [
            pair
            for pair in label_one
            if pair[0] in members
            and pair[1] in members
            and pair[0] not in links
            and pair[1] not in links
        ]
        selected = self._select_diverse(pool, budget)
        return [CategorizedPair(tuple(pair), EXTRA_MATCH) for pair in selected]

    def _negatives(self, records: list[RefId], positives: int) -> list[CategorizedPair]:
        count = min(
            round(self.neg_pos_ratio * positives), self._cross_cluster_pairs(records)
        )
        if count <= 0:
            return []
        pairs = self._generate_negatives(count, set(self._matcher_gt), records)
        return [CategorizedPair(pair, NEGATIVE) for pair in pairs]

    def _cross_cluster_pairs(self, records: list[RefId]) -> int:
        sizes = Counter(self._cluster_gt[r] for r in records)
        n = len(records)
        return (n * n - sum(s * s for s in sizes.values())) // 2

    def _check_invariants(
        self, split_of: dict[int, str], links: Links, sources: Links
    ) -> None:
        rows = {name: _frame_rows(frame) for name, frame in self._splits.items()}
        for name, split_rows in rows.items():
            self._check_disjoint(name, split_rows, split_of)
            self._check_footprints(name, split_rows, split_of, links)
        self._check_label_two(rows, split_of, links)
        self._check_sources_kept(rows, split_of, links, sources)
        self._warn_scarce_label_two(rows)

    def _check_disjoint(
        self, name: str, rows: list[Row], split_of: dict[int, str]
    ) -> None:
        for left, right, _ in rows:
            splits = {split_of[self._cluster_gt[ref]] for ref in (left, right)}
            if len(splits) > 1:
                _violation(f"row ({left}, {right}) spans {sorted(splits)}")
            if splits != {name}:
                _violation(f"row ({left}, {right}) in {name} belongs to {splits}")

    def _check_footprints(
        self,
        name: str,
        rows: list[Row],
        split_of: dict[int, str],
        links: Links,
    ) -> None:
        mentioned = {ref for left, right, _ in rows for ref in (left, right)}
        bridges = sorted(mentioned & links.keys(), key=_ref_key)
        if orphans := [b for b in bridges if b in self.orphan_records]:
            _violation(f"orphan bridges {orphans} appear in {name}")
        for bridge in bridges:
            footprint = self._footprint(bridge, links[bridge])
            if outside := [c for c in footprint if split_of[c] != name]:
                _violation(
                    f"{bridge} in {name}: footprint clusters {outside} elsewhere"
                )

    def _check_label_two(
        self,
        rows: dict[str, list[Row]],
        split_of: dict[int, str],
        links: Links,
    ) -> None:
        present = {name: set(split_rows) for name, split_rows in rows.items()}
        for bridge, pairs in links.items():
            if bridge in self.orphan_records:
                continue
            name = split_of[self._cluster_gt[bridge]]
            if missing := [p for p in pairs if (*p, 2) not in present[name]]:
                _violation(f"{bridge} in {name}: label-2 pairs {missing} missing")

    def _check_sources_kept(
        self,
        rows: dict[str, list[Row]],
        split_of: dict[int, str],
        links: Links,
        sources: Links,
    ) -> None:
        present = {name: set(split_rows) for name, split_rows in rows.items()}
        for bridge in links:
            if bridge in self.orphan_records:
                continue
            name = split_of[self._cluster_gt[bridge]]
            kept = sum(1 for p in sources[bridge] if (*p, 1) in present[name])
            if kept != (quota := self._source_quota(bridge, links)):
                _violation(f"{bridge} in {name}: {kept} triplet sources, not {quota}")

    def _warn_scarce_label_two(self, rows: dict[str, list[Row]]) -> None:
        for name, split_rows in rows.items():
            count = sum(1 for *_, label in split_rows if label == 2)
            if count < MIN_PER_CLASS:
                self._log.warning(
                    "split %s has %d label-2 pairs, fewer than MIN_PER_CLASS=%d",
                    name,
                    count,
                    MIN_PER_CLASS,
                )
