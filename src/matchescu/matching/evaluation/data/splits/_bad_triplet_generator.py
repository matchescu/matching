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
    """Group the clusters that must share a split because bridges join them.

    A bridge's text derives from every cluster of its footprint, so splitting a
    component would leak that text across splits. Components are therefore the
    smallest units the generator may assign to a split.

    :param clusters: Ids of the component's clusters, ascending.
    :param bridges: Bridge records whose footprints form the component, sorted.
    :param label_two: Label-2 pairs of those bridges; the component's weight when
        balancing label-2 pairs across splits.
    """

    clusters: tuple[int, ...]
    bridges: tuple[RefId, ...]
    label_two: int


class CategorizedPair(NamedTuple):
    """Tag a split row with its role in a bad triplet, for the split metadata.

    The split CSVs only carry the class label, which cannot tell a triplet source
    from an extra match. Keeping the role and the bridge of every row lets
    `BadTripletSplitGenerator.save` write meta files, so that evaluations can
    report results per bad triplet rather than per class.

    :param pair: Compared records, oriented as in the matcher ground truth.
    :param category: One of ``negative``, ``triplet_source``, ``extra_match`` or
        ``triplet_target``.
    :param bridge: Bridge record of a triplet row; ``None`` for negatives and
        extra matches.
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
    """Split bad-triplet data into train / dev / test without leaking bridge text.

    A bridge record resembles clusters it does not belong to. If any of those
    clusters lands in another split, the model sees the bridge's text during
    training and evaluation alike, which inflates evaluation scores. This
    generator prevents that:

    - a bridge's footprint is its home cluster plus its target clusters;
    - bridges join their footprints into components; each component goes whole
      to one split, largest first, wherever label-2 pairs are most lacking;
    - every split must receive label-2 pairs;
    - every bridge keeps all its label-2 pairs and a random sample of label-1
      pairs from its home cluster, so the bad triplet stays complete;
    - label-1 pairs without a bridge fill the rest of the label-1 budget;
    - negatives never repeat a known pair and, by default, never join a bridge
      with its target clusters, whose resemblance is deliberate.

    .. note::
       A cluster may be the home of one bridge at most. Several bridges per home
       would make one cluster ambiguous towards several others, which the bad
       triplet model does not cover.

    :param drop_orphans: Kept for backward compatibility. Orphan bridges are
        always dropped; ``False`` only logs a warning.
    :param source_links_per_bridge: Upper bound on the label-1 pairs kept per
        bridge. Defaults to ``None``: keep as many as ``match_bridge_ratio`` and
        the home cluster size allow.
    :param allow_target_negatives: Let negatives join a bridge with members of its
        target clusters. Defaults to ``False``.
    :ivar orphan_records: Dropped bridges mapped to the reason they were dropped.
        Filled by `generate`.
    """

    def __init__(
        self,
        *args,
        drop_orphans: bool = True,
        source_links_per_bridge: int | None = None,
        allow_target_negatives: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not drop_orphans:
            self._log.warning(
                "drop_orphans=False is ignored: orphan bridges are always dropped"
            )
        self.source_links_per_bridge = source_links_per_bridge
        self.allow_target_negatives = allow_target_negatives
        self.orphan_records: dict[RefId, str] = {}
        self._select_bridges: BridgeSelector = _label_two_right_records
        self._categorized: dict[str, list[CategorizedPair]] = {}

    def with_bridge_selector(self, selector: BridgeSelector) -> Self:
        """Replace the rule that recognises bridge records.

        Bridges are the right records of label-2 pairs by default. Data sources
        that mark bridges differently, for example by their record source, can
        plug in their own rule here without subclassing.

        :param selector: Receives the matcher ground truth and returns the bridge
            records. Every right record of a label-2 pair must be among them.
        :returns: This generator, to allow chaining before `generate`.
        """
        self._select_bridges = selector
        return self

    def generate(self) -> Self:
        """Assign every cluster to a split and build the labelled rows of each split.

        Clusters, not pairs, are assigned so that no record appears in two
        splits. The result is verified before returning, so a returned generator
        always holds leak-free splits.

        :returns: This generator, ready for `get_splits` or `save`.
        :raises RuntimeError: If `load` was not called, or if the built splits
            break a leak-freedom guarantee (an internal error).
        :raises ValueError: If the ground truth has classes other than 0, 1 and 2;
            a label-2 pair's right record is not a bridge, lacks a cluster or
            shares its target's cluster; a cluster is home to several bridges; a
            bridge has no label-1 pair in its home cluster; or a split would
            receive no label-2 pairs.
        """
        self._check_loaded()
        # validate the ground truth classes
        by = self._pairs_by_class()
        if unsupported := by.keys() - {0, 1, 2}:
            raise ValueError(
                f"bad triplet split does not support classes {unsupported}"
            )
        # find each bridge's label-2 and label-1 pairs
        links = self._bridge_links()
        sources = self._source_pairs(links)
        self._check_sources(links, sources)
        # group the clusters that must share a split
        components = self._components(links)
        self._log_components(components)
        # assign clusters to splits, then build each split's rows
        split_of = self._assign(components, links)
        self.orphan_records = {}
        self._splits = self._build_splits(by.get(1, []), split_of, links, sources)
        # verify that no bridge text leaks across splits
        self._check_invariants(split_of, links, sources)
        return self

    def save(
        self,
        output_dir: str | os.PathLike,
        prefix: str = "",
        file_names: Mapping[str, str] | None = None,
    ) -> Self:
        """Write the split CSV files and one metadata file per split.

        The split files keep the schema every matcher reads. The extra
        ``{prefix}_{split}_meta.csv`` files list the same rows in the same
        order, adding each row's category and bridge, so evaluations can group
        results by bad triplet.

        :param output_dir: Directory receiving all files; created if missing.
        :param prefix: Prepended to every file name. Defaults to no prefix.
        :param file_names: Custom names for the split files, by split name.
            Meta files always use the prefix pattern. Defaults to ``None``.
        :returns: This generator.
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
        # sorted, so that the result does not depend on hash order
        for pair in sorted(label_two, key=_pair_key):
            self._check_link(pair, bridges)
            links[pair[1]].append(pair)
        # one bridge per home cluster
        self._check_homes(links)
        return dict(links)

    def _check_homes(self, links: Links) -> None:
        homes = Counter(self._cluster_gt[bridge] for bridge in links)
        if shared := sorted(cluster for cluster, n in homes.items() if n > 1):
            raise ValueError(f"clusters home to several bridges: {shared}")

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
        # 1. whole components, balancing label-2 pairs
        label_two = self._assign_components(components, split_of)
        # 2. clusters with label-1 pairs, in proportion to each split's label-2
        self._assign_match_clusters(self._match_counts(links), label_two, split_of)
        # 3. the remaining clusters, balancing record counts
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
        # positives first: their count sets the number of negatives
        positives = self._positives(name, set(records), label_one, links, sources)
        # negatives from this split's records, avoiding excluded pairs
        exclude = self._negative_exclusions(name, records, links)
        rows = self._negatives(records, len(positives), exclude) + positives
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
        # every label-2 pair of every bridge in the split
        targets = [
            CategorizedPair(p, TRIPLET_TARGET, b) for b in bridges for p in links[b]
        ]
        # a random sample of each bridge's label-1 pairs in its home cluster
        anchored = [
            CategorizedPair(p, TRIPLET_SOURCE, b)
            for b in bridges
            for p in self._anchor(b, sources[b], links)
        ]
        # label-1 pairs without a bridge fill the remaining label-1 budget
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

    def _negative_exclusions(
        self, name: str, records: list[RefId], links: Links
    ) -> set[Pair]:
        excluded = self._target_pairs(records, links)
        self._log.info(
            "split %s: %d bridge-target candidate pairs excluded from negatives",
            name,
            len(excluded),
        )
        return set(self._matcher_gt) | excluded

    def _target_pairs(self, records: list[RefId], links: Links) -> set[Pair]:
        if self.allow_target_negatives:
            return set()
        members = set(records)
        return {
            (other, bridge)
            for bridge, pairs in links.items()
            if bridge in members
            for cluster in self._footprint(bridge, pairs)
            if cluster != self._cluster_gt[bridge]
            for other in self._clusters[cluster]
        }

    def _negatives(
        self, records: list[RefId], positives: int, exclude: set[Pair]
    ) -> list[CategorizedPair]:
        count = min(
            round(self.neg_pos_ratio * positives), self._cross_cluster_pairs(records)
        )
        if count <= 0:
            return []
        pairs = self._generate_negatives(count, exclude, records)
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
            # each row lies within its clusters' split
            self._check_disjoint(name, split_rows, split_of)
            # each bridge's footprint lies within the bridge's split
            self._check_footprints(name, split_rows, split_of, links)
        # each kept bridge keeps all its label-2 pairs
        self._check_label_two(rows, split_of, links)
        # each kept bridge keeps its quota of home label-1 pairs
        self._check_sources_kept(rows, split_of, links, sources)
        # no negative repeats a known pair or a bridge-target pair
        self._check_negatives(rows, links)
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

    def _check_negatives(self, rows: dict[str, list[Row]], links: Links) -> None:
        forbidden = set(self._matcher_gt) | self._target_pairs(
            list(self._cluster_gt), links
        )
        for name, split_rows in rows.items():
            for left, right, label in split_rows:
                if label == 0 and {(left, right), (right, left)} & forbidden:
                    _violation(f"negative ({left}, {right}) in {name} is excluded")

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
