import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple, NoReturn, Self

import polars as pl
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching.evaluation.data.sampling import BadTripletSampler

from ._split import Split
from ._split_generator import MIN_PER_CLASS, SplitGenerator, _pairs_frame

Pair = tuple[RefId, RefId]
Row = tuple[RefId, RefId, int]
Unit = tuple[tuple[int, ...], int]

NEGATIVE = "negative"
TRIPLET_SOURCE = "triplet_source"
EXTRA_MATCH = "extra_match"
TRIPLET_TARGET = "triplet_target"
MAX_NEGATIVE_ROUNDS = 5


class Component(NamedTuple):
    """Group the clusters that must share a split because bridges join them.

    A bridge's text derives from every cluster of its footprint, so splitting a
    component would leak that text across splits. Components are therefore the
    smallest units the generator may assign to a split.

    :param clusters: Ids of the component's clusters, ascending.
    :param bridges: Bridge records whose footprints form the component, sorted.
    :param link_count: Bridge-link pairs of those bridges; the component's
        weight when balancing link rows across splits.
    """

    clusters: tuple[int, ...]
    bridges: tuple[RefId, ...]
    link_count: int


class CategorizedPair(NamedTuple):
    """Tag a split row with its role in a bad triplet, for the split metadata.

    The split CSVs only carry the class label, which cannot tell a triplet source
    from an extra match. Keeping the role and the bridge of every row lets
    `BadTripletSplitGenerator.save` write meta files, so that evaluations can
    report results per bad triplet rather than per class.

    :param pair: Compared records, with any bridge on the right.
    :param category: One of ``negative``, ``triplet_source``, ``extra_match`` or
        ``triplet_target``.
    :param bridge: Bridge record of a triplet row; ``None`` for negatives and
        extra matches.
    :param label: Label of the pair in the matcher ground truth. Bridge links
        carry label 2 in directed ground truths and label 1 in binary ones.
    """

    pair: Pair
    category: str
    bridge: RefId | None = None
    label: int = 0


def _ref_key(ref: RefId) -> tuple[str, str]:
    return ref.source, str(ref.label)


def _counterpart(pair: Pair, ref: RefId) -> RefId:
    left, right = pair
    return right if left == ref else left


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


def _labelled_frame(rows: list[CategorizedPair]) -> pl.DataFrame:
    samples: dict[int, list[Pair]] = defaultdict(list)
    for row in rows:
        samples[row.label].append(row.pair)
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
      to one split, largest first, wherever link rows are most lacking;
    - every split must receive bridge links;
    - every bridge keeps one triplet source per link, capped by the home pairs
      available to it and sampled round-robin across the bridges that share a
      home cluster;
    - bridge links are always written with the bridge on the right, whatever
      the labelling scheme of the ground truth;
    - each bridge link receives ``match_bridge_ratio`` full-match rows: one
      triplet source and the rest legitimate matches from the bridge's home
      cluster that do not involve any bridge. When the home cluster cannot
      supply them, the split's other clusters supply the remainder; as a last
      resort, home pairs of other bridges are consumed, which invalidates those
      bridges. A split that still misses the ratio raises;
    - negatives never repeat a known pair and, by default, never join a bridge
      with its target clusters, whose resemblance is deliberate;
    - a warning flags components too large for the smallest split.

    .. note::
       Bridge records are declared by `is_bridge`, defaulting to the right
       records of label-2 pairs. Binary ground truths mark bridges by another
       rule, such as the record source. Pairs between two bridges are ignored:
       they belong to no triplet. Orphan bridges are treated as a data error.

    :param is_bridge: Predicate that marks bridge records. Defaults to ``None``,
        which derives the predicate from label-2 pairs; binary ground truths
        must supply their own rule.
    :param allow_target_negatives: Let negatives join a bridge with members of
        its target clusters. Defaults to ``False``.
    """

    def __init__(
        self,
        *args,
        is_bridge: Callable[[RefId], bool] | None = None,
        allow_target_negatives: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.is_bridge = is_bridge
        self.allow_target_negatives = allow_target_negatives
        self._categorized: dict[str, list[CategorizedPair]] = {}
        self._bridges: frozenset[RefId] = frozenset()
        self._targets: dict[RefId, tuple[RefId, ...]] = {}
        self._homes: dict[RefId, tuple[RefId, ...]] = {}
        self._extra_pools: dict[int, list[Pair]] = {}
        self._split_state: dict[str, dict[RefId, tuple[list[RefId], list[Pair]]]] = {}

    def generate(self) -> Self:
        """Assign every cluster to a split and build the labelled rows of each split.

        Clusters, not pairs, are assigned so that no record appears in two
        splits. The result is verified before returning, so a returned generator
        always holds leak-free splits.

        :returns: This generator, ready for `get_splits` or `save`.
        :raises RuntimeError: If `load` was not called, or if the built splits
            break a leak-freedom guarantee (an internal error).
        :raises ValueError: If the ground truth has classes other than 0, 1 and
            2, the bridge structure is malformed, a bridge is orphaned, or a
            split cannot reach the match-bridge ratio.
        """
        self._check_loaded()
        self._check_classes()
        index = self._index()
        self._validate_index(index)
        self._load_structure(index)
        components = self._components()
        self._log_components(components)
        split_of = self._assign(components)
        self._split_state = {}
        self._splits = self._build_splits(split_of)
        self._check_invariants(split_of)
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
        :param prefix: Prepended to every file name. Defaults to none.
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

    def get_splits(self) -> dict[str, Split]:
        """Return the generated splits as comparison spaces with ground truth.

        Label lookup tolerates either orientation because this generator writes
        bridge rows with the bridge on the right, which may reverse the
        orientation recorded in the ground truth.

        :returns: The generated splits, keyed by split name.
        :raises RuntimeError: If `generate` was not called first.
        """
        if not self._splits:
            raise RuntimeError("generate() before returning splits")
        splits = {}
        for split_name, frame in self._splits.items():
            pairs = [
                (
                    RefId(row["left_id"], row["left_source"]),
                    RefId(row["right_id"], row["right_source"]),
                )
                for row in frame.iter_rows(named=True)
            ]
            comparison_space = InMemoryComparisonSpace()
            match_gt: dict[Pair, int] = {}
            cluster_gt: dict[RefId, int] = {}
            for x, y in pairs:
                comparison_space.put(x, y)
                label = self._label_of((x, y))
                if label == 0:
                    continue
                match_gt[(x, y)] = label
                for ref in (x, y):
                    cluster = self._cluster_gt.get(ref)
                    if cluster is not None:
                        cluster_gt[ref] = cluster
            clusters = defaultdict(set)
            for ref, cluster in cluster_gt.items():
                clusters[cluster].add(ref)
            splits[split_name] = Split(comparison_space, match_gt, clusters)
        return splits

    def _check_classes(self) -> None:
        classes = set(self._matcher_gt.values())
        if unsupported := classes - {0, 1, 2}:
            raise ValueError(
                f"bad triplet split does not support classes {unsupported}"
            )

    def _index(self):
        predicate = self.is_bridge or self._label_two_predicate()
        return BadTripletSampler(
            self._id_table,
            self._matcher_gt,
            self._cluster_gt,
            predicate,
            log=self._log,
        ).index()

    def _label_two_predicate(self) -> Callable[[RefId], bool]:
        bridges = frozenset(
            right for (_, right), label in self._matcher_gt.items() if label == 2
        )
        return lambda ref: ref in bridges

    def _validate_index(self, index) -> None:
        fatal = [issue for issue in index.issues if issue[1] != "bridge_bridge"]
        if fatal:
            raise ValueError(f"invalid bridge structure: {fatal}")
        if ignored := [issue for issue in index.issues if issue[1] == "bridge_bridge"]:
            self._log.info("ignoring %d bridge-bridge pairs", len(ignored))
        if index.orphan_bridges:
            raise ValueError(f"orphan bridges: {index.orphan_bridges}")
        if not index.link_pairs:
            raise ValueError(
                "no bridge links declared; binary ground truths must pass " "is_bridge"
            )

    def _load_structure(self, index) -> None:
        self._bridges = frozenset(index.link_pairs)
        self._targets = {
            bridge: tuple(_counterpart(pair, bridge) for pair in pairs)
            for bridge, pairs in index.link_pairs.items()
        }
        self._homes = {
            bridge: tuple(_counterpart(pair, bridge) for pair in pairs)
            for bridge, pairs in index.home_pairs.items()
        }
        self._extra_pools = self._full_match_pools()

    def _full_match_pools(self) -> dict[int, list[Pair]]:
        pools: dict[int, list[Pair]] = defaultdict(list)
        seen: set[frozenset] = set()
        for pair, label in self._matcher_gt.items():
            if not label:
                continue
            left, right = pair
            if left in self._bridges or right in self._bridges:
                continue
            cluster = self._cluster_gt.get(left)
            if cluster is None or cluster != self._cluster_gt.get(right):
                continue
            key = frozenset(pair)
            if key in seen:
                continue
            seen.add(key)
            pools[cluster].append(pair)
        return dict(pools)

    def _footprint(self, bridge: RefId) -> list[int]:
        targets = {self._cluster_gt[target] for target in self._targets[bridge]}
        return sorted(targets | {self._cluster_gt[bridge]})

    def _components(self) -> list[Component]:
        parent: dict[int, int] = {}
        for bridge in self._targets:
            _union(parent, self._footprint(bridge))
        clusters: dict[int, list[int]] = defaultdict(list)
        for cluster in sorted(parent):
            clusters[_find(parent, cluster)].append(cluster)
        bridges: dict[int, list[RefId]] = defaultdict(list)
        for bridge in self._targets:
            bridges[_find(parent, self._cluster_gt[bridge])].append(bridge)
        components = (
            Component(
                tuple(clusters[root]),
                tuple(members),
                sum(len(self._targets[b]) for b in members),
            )
            for root, members in bridges.items()
        )
        return sorted(components, key=lambda component: component.clusters[0])

    def _ratio_shares(self, total: float) -> dict[str, float]:
        weight = sum(self.split_ratio.values())
        return {name: total * r / weight for name, r in self.split_ratio.items()}

    def _log_components(self, components: list[Component]) -> None:
        sizes = [component.link_count for component in components]
        total = sum(sizes)
        self._log.info("bridge graph: %d components", len(components))
        self._log.info(
            "bridge links per component: %s", dict(sorted(Counter(sizes).items()))
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
            "split link targets: %s",
            ", ".join(f"{k}={v:.1f}" for k, v in targets.items()),
        )
        if sizes and max(sizes) > (smallest := min(targets.values())):
            self._log.warning(
                "largest component has %d bridge links and exceeds the smallest "
                "split target %.1f",
                max(sizes),
                smallest,
            )

    def _assign(self, components: list[Component]) -> dict[int, str]:
        split_of: dict[int, str] = {}
        links = self._assign_components(components, split_of)
        self._assign_match_clusters(self._extra_counts(), links, split_of)
        self._assign_remaining(split_of)
        return split_of

    def _assign_components(
        self, components: list[Component], split_of: dict[int, str]
    ) -> dict[str, float]:
        ordered = sorted(components, key=lambda c: (-c.link_count, c.clusters[0]))
        targets = self._ratio_shares(sum(c.link_count for c in components))
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        units = [(c.clusters, c.link_count) for c in ordered]
        self._place_greedy(units, targets, assigned, split_of)
        self._log.info(
            "split links assigned: %s",
            ", ".join(f"{k}={assigned[k]:.0f}/{v:.1f}" for k, v in targets.items()),
        )
        if empty := [name for name, count in assigned.items() if not count]:
            raise ValueError(f"splits without bridge links: {empty}")
        return assigned

    def _extra_counts(self) -> Counter[int]:
        counts: Counter[int] = Counter()
        for cluster, pairs in self._extra_pools.items():
            counts[cluster] += len(pairs)
        return counts

    def _assign_match_clusters(
        self,
        counts: Counter[int],
        links: dict[str, float],
        split_of: dict[int, str],
    ) -> None:
        assigned = dict.fromkeys(self.split_ratio, 0.0)
        for cluster, split in split_of.items():
            assigned[split] += counts[cluster]
        free = sorted(
            (c for c in counts if c not in split_of), key=lambda c: (-counts[c], c)
        )
        targets = self._match_targets(counts.total(), links)
        self._place_greedy(
            [((c,), counts[c]) for c in free], targets, assigned, split_of
        )

    @staticmethod
    def _match_targets(total: int, links: dict[str, float]) -> dict[str, float]:
        received = sum(links.values())
        return {name: total * n / received for name, n in links.items()}

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

    def _build_splits(self, split_of: dict[int, str]) -> dict[str, pl.DataFrame]:
        self._categorized = {
            name: self._compose(name, split_of) for name in self.split_ratio
        }
        return {name: _labelled_frame(rows) for name, rows in self._categorized.items()}

    def _compose(self, name: str, split_of: dict[int, str]) -> list[CategorizedPair]:
        bridges = [b for b in self._targets if split_of[self._cluster_gt[b]] == name]
        if not bridges:
            raise ValueError(f"split {name}: no bridge links")
        sources = self._pick_sources(bridges)
        extras, active = self._pick_extras(name, split_of, bridges, sources)
        self._split_state[name] = {
            bridge: (sources[bridge], extras.get(bridge, [])) for bridge in active
        }
        rows: list[CategorizedPair] = []
        for bridge in active:
            rows.extend(
                self._bridge_rows(bridge, sources[bridge], extras.get(bridge, []))
            )
        records = [ref for ref, c in self._cluster_gt.items() if split_of[c] == name]
        rows.extend(self._negatives(records, len(rows)))
        self._log.info(
            "split %s: %d records, %s",
            name,
            len(records),
            dict(sorted(Counter(row.category for row in rows).items())),
        )
        return rows

    def _bridge_rows(
        self, bridge: RefId, sources: list[RefId], extras: list[Pair]
    ) -> list[CategorizedPair]:
        rows = []
        for target in self._targets[bridge]:
            pair = (target, bridge)
            rows.append(
                CategorizedPair(pair, TRIPLET_TARGET, bridge, self._label_of(pair))
            )
        for home in sources:
            pair = (home, bridge)
            rows.append(
                CategorizedPair(pair, TRIPLET_SOURCE, bridge, self._label_of(pair))
            )
        for pair in extras:
            rows.append(CategorizedPair(pair, EXTRA_MATCH, None, self._label_of(pair)))
        return rows

    def _label_of(self, pair: Pair) -> int:
        if pair in self._matcher_gt:
            return self._matcher_gt[pair]
        return self._matcher_gt.get((pair[1], pair[0]), 0)

    def _pick_sources(self, bridges: list[RefId]) -> dict[RefId, list[RefId]]:
        groups: dict[int, list[RefId]] = defaultdict(list)
        for bridge in bridges:
            groups[self._cluster_gt[bridge]].append(bridge)
        sources: dict[RefId, list[RefId]] = {}
        for group in groups.values():
            sources.update(self._sample_sources(group))
        return sources

    def _sample_sources(self, group: list[RefId]) -> dict[RefId, list[RefId]]:
        pools = {}
        for bridge in group:
            pool = list(self._homes.get(bridge, ()))
            self._rng.shuffle(pool)
            pools[bridge] = pool
        remaining = {bridge: len(self._targets[bridge]) for bridge in group}
        counters = dict.fromkeys(group, 0)
        chosen: dict[RefId, dict[RefId, None]] = {bridge: {} for bridge in group}
        while any(remaining.values()):
            order = [bridge for bridge in group if remaining[bridge] > 0]
            self._rng.shuffle(order)
            for bridge in order:
                if remaining[bridge] <= 0:
                    continue
                pool = pools[bridge]
                if pool:
                    home = pool[counters[bridge] % len(pool)]
                    counters[bridge] += 1
                    chosen[bridge][home] = None
                remaining[bridge] -= 1
        return {bridge: list(homes) for bridge, homes in chosen.items()}

    def _pick_extras(
        self,
        name: str,
        split_of: dict[int, str],
        bridges: list[RefId],
        sources: dict[RefId, list[RefId]],
    ) -> tuple[dict[RefId, list[Pair]], list[RefId]]:
        needs = {
            bridge: round(self.match_bridge_ratio * len(self._targets[bridge]))
            - len(sources[bridge])
            for bridge in bridges
        }
        if missing := {b: n for b, n in needs.items() if n < 0}:
            raise ValueError(f"split {name}: negative extra-match budget {missing}")
        extras: dict[RefId, list[Pair]] = defaultdict(list)
        used: set[frozenset] = set()
        self._fill_home_extras(bridges, needs, extras, used)
        self._fill_split_extras(name, split_of, bridges, needs, extras, used)
        active = self._fill_bridge_extras(name, bridges, needs, extras, used)
        if invalidated := len(bridges) - len(active):
            self._log.warning(
                "split %s: %d/%d bridges invalidated while filling the "
                "match-bridge ratio",
                name,
                invalidated,
                len(bridges),
            )
        if outstanding := {b: n for b, n in needs.items() if n > 0}:
            raise ValueError(
                f"split {name}: match-bridge ratio not met for {outstanding}"
            )
        return extras, active

    def _fill_home_extras(self, bridges, needs, extras, used) -> None:
        groups: dict[int, list[RefId]] = defaultdict(list)
        for bridge in bridges:
            groups[self._cluster_gt[bridge]].append(bridge)
        for cluster, group in groups.items():
            candidates = list(self._extra_pools.get(cluster, ()))
            self._rng.shuffle(candidates)
            self._assign_pairs(candidates, group, needs, extras, used)

    def _fill_split_extras(self, name, split_of, bridges, needs, extras, used) -> None:
        candidates = []
        for cluster, split in split_of.items():
            if split == name:
                candidates.extend(self._extra_pools.get(cluster, ()))
        self._rng.shuffle(candidates)
        self._assign_pairs(candidates, bridges, needs, extras, used)

    def _assign_pairs(self, candidates, bridges, needs, extras, used) -> None:
        index = 0
        while index < len(candidates) and any(needs[b] > 0 for b in bridges):
            order = [bridge for bridge in bridges if needs[bridge] > 0]
            self._rng.shuffle(order)
            for bridge in order:
                if index >= len(candidates):
                    break
                pair = candidates[index]
                index += 1
                key = frozenset(pair)
                if key in used:
                    continue
                used.add(key)
                extras[bridge].append(pair)
                needs[bridge] -= 1

    def _fill_bridge_extras(self, name, bridges, needs, extras, used) -> list[RefId]:
        active = list(bridges)
        while active and any(needs[b] > 0 for b in active):
            needy = [bridge for bridge in active if needs[bridge] > 0]
            pair, owner = self._draw_bridge_pair(bridges, needy, used)
            if pair is None:
                return active
            options = [bridge for bridge in needy if bridge != owner]
            chosen = options[self._rng.randint(len(options))]
            used.add(frozenset(pair))
            extras[chosen].append(pair)
            needs[chosen] -= 1
            if owner in active:
                active = self._invalidate(name, owner, active, needs, extras, used)
        return active

    def _draw_bridge_pair(self, bridges, needy, used):
        candidates = [
            ((home, owner), owner)
            for owner in bridges
            for home in self._homes.get(owner, ())
            if frozenset((home, owner)) not in used
        ]
        self._rng.shuffle(candidates)
        for pair, owner in candidates:
            if any(bridge != owner for bridge in needy):
                return pair, owner
        return None, None

    def _invalidate(self, name, owner, active, needs, extras, used) -> list[RefId]:
        remaining = [bridge for bridge in active if bridge != owner]
        needs.pop(owner, None)
        for pair in extras.pop(owner, []):
            used.discard(frozenset(pair))
        if not remaining:
            raise ValueError(
                f"split {name}: every bridge was invalidated while filling "
                "the match-bridge ratio"
            )
        return remaining

    def _negatives(self, records: list[RefId], positives: int) -> list[CategorizedPair]:
        count = min(
            round(self.neg_pos_ratio * positives), self._cross_cluster_pairs(records)
        )
        if count <= 0:
            return []
        blocked = set(self._matcher_gt) | self._target_pairs(records)
        pairs: list[Pair] = []
        for _ in range(MAX_NEGATIVE_ROUNDS):
            missing = count - len(pairs)
            if missing <= 0:
                break
            batch = self._generate_negatives(missing, blocked, records)
            if not batch:
                break
            for pair in batch:
                blocked.add(pair)
                blocked.add((pair[1], pair[0]))
                if pair[0] in self._bridges and pair[1] in self._bridges:
                    continue
                pairs.append(self._orient(pair))
        if len(pairs) < count:
            self._log.warning("generated %d/%d negatives", len(pairs), count)
        return [CategorizedPair(pair, NEGATIVE) for pair in pairs]

    def _orient(self, pair: Pair) -> Pair:
        left, right = pair
        if left in self._bridges and right not in self._bridges:
            return right, left
        return pair

    def _target_pairs(self, records: list[RefId]) -> set[Pair]:
        if self.allow_target_negatives:
            return set()
        members = set(records)
        return {
            (other, bridge)
            for bridge in self._targets
            if bridge in members
            for cluster in self._footprint(bridge)
            if cluster != self._cluster_gt[bridge]
            for other in self._clusters[cluster]
        }

    def _cross_cluster_pairs(self, records: list[RefId]) -> int:
        sizes = Counter(self._cluster_gt[r] for r in records)
        n = len(records)
        return (n * n - sum(s * s for s in sizes.values())) // 2

    def _check_invariants(self, split_of: dict[int, str]) -> None:
        rows = self._categorized
        for name, split_rows in rows.items():
            self._check_disjoint(name, split_rows, split_of)
            self._check_bridge_position(name, split_rows)
            self._check_unique(name, split_rows)
            self._check_links(name, split_rows)
            self._check_sources(name, split_rows)
            self._check_ratio(name, split_rows)
            self._check_negatives(name, split_rows)
            self._warn_scarce_links(name, split_rows)

    def _check_disjoint(
        self, name: str, rows: list[CategorizedPair], split_of: dict[int, str]
    ) -> None:
        for row in rows:
            splits = {
                split_of[self._cluster_gt[ref]]
                for ref in row.pair
                if ref in self._cluster_gt
            }
            if len(splits) > 1:
                _violation(f"row {row.pair} spans {sorted(splits)}")
            if splits and splits != {name}:
                _violation(f"row {row.pair} in {name} belongs to {splits}")

    def _check_bridge_position(self, name: str, rows: list[CategorizedPair]) -> None:
        for row in rows:
            left, _ = row.pair
            if left in self._bridges:
                _violation(f"{name}: bridge {left} is on the left of {row.pair}")

    def _check_unique(self, name: str, rows: list[CategorizedPair]) -> None:
        seen: set[frozenset] = set()
        for row in rows:
            key = frozenset(row.pair)
            if key in seen:
                _violation(f"{name}: duplicate pair {row.pair}")
            seen.add(key)

    def _check_links(self, name: str, rows: list[CategorizedPair]) -> None:
        present = {row.pair for row in rows if row.category == TRIPLET_TARGET}
        for bridge in {row.bridge for row in rows if row.category == TRIPLET_TARGET}:
            expected = {(target, bridge) for target in self._targets[bridge]}
            if present & expected != expected:
                _violation(f"{name}: bridge {bridge} misses link rows")

    def _check_sources(self, name: str, rows: list[CategorizedPair]) -> None:
        present = {
            (row.bridge, row.pair) for row in rows if row.category == TRIPLET_SOURCE
        }
        for bridge, (sources, _) in self._split_state[name].items():
            expected = {(bridge, (home, bridge)) for home in sources}
            if present & expected != expected:
                _violation(f"{name}: bridge {bridge} misses triplet source rows")

    def _check_ratio(self, name: str, rows: list[CategorizedPair]) -> None:
        active = self._split_state[name]
        for row in rows:
            if row.category != EXTRA_MATCH:
                continue
            for ref in row.pair:
                if ref in self._bridges and ref in active:
                    _violation(
                        f"{name}: extra match {row.pair} involves active bridge {ref}"
                    )
        for bridge, (sources, extras) in active.items():
            budget = round(self.match_bridge_ratio * len(self._targets[bridge]))
            if len(sources) + len(extras) != budget:
                _violation(
                    f"{name}: bridge {bridge} has {len(sources) + len(extras)} "
                    f"full matches, expected {budget}"
                )

    def _check_negatives(self, name: str, rows: list[CategorizedPair]) -> None:
        forbidden = set(self._matcher_gt) | self._target_pairs(list(self._cluster_gt))
        for row in rows:
            if row.label != 0:
                continue
            left, right = row.pair
            if {(left, right), (right, left)} & forbidden:
                _violation(f"{name}: negative {row.pair} is excluded")
            if left in self._bridges and right not in self._bridges:
                _violation(f"{name}: misoriented bridge negative {row.pair}")
            if left in self._bridges and right in self._bridges:
                _violation(f"{name}: bridge pair {row.pair}")

    def _warn_scarce_links(self, name: str, rows: list[CategorizedPair]) -> None:
        count = sum(1 for row in rows if row.category == TRIPLET_TARGET)
        if count < MIN_PER_CLASS:
            self._log.warning(
                "split %s has %d bridge links, fewer than MIN_PER_CLASS=%d",
                name,
                count,
                MIN_PER_CLASS,
            )
