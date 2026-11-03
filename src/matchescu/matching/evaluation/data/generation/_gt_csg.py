"""Generate comparison spaces from entity-resolution ground truths.

The generator supports two kinds of datasets. A dataset with bridge records
produces a comparison space built around bad triplets: every bridge link
brings one triplet source and enough full-match contrast rows to reach
``match_bridge_ratio`` full matches per link, plus negatives at
``neg_pos_ratio``. A dataset without bridge records falls back to the classic
class-balanced sampling, which is the only meaningful composition when no
triplets exist.

Exclusions drive both modes. A pair without a bridge record is a hard
exclusion and never enters the comparison space. A pair with a bridge record
avoids that bridge as a whole and bends under quota pressure, at which point
its triplets become usable. Exclusions resolved by the bad-triplet sampler
also name bridges that were demoted to exact-pair exclusions.

Bridge records are declared through a predicate, so directed ground truths
mark them by label 2 and binary ground truths by another rule such as the
record source.
"""

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from os import PathLike
from typing import ClassVar

import numpy as np
import polars as pl
from matchescu.reference_store.comparison_space import (
    BinaryComparisonSpace,
    InMemoryComparisonSpace,
)
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching._logging import resolve_logger
from matchescu.matching.evaluation.data.sampling import (
    BadTripletSample,
    BadTripletSampler,
    BridgeIndex,
)
from matchescu.matching.evaluation.ground_truth import EquivalenceClassPartitioner

Pair = tuple[RefId, RefId]


class GroundTruthComparisonSpaceGenerator:
    """Build a comparison space from a matching ground truth.

    The generator exists to produce evaluation data with controlled positive
    and negative composition. With bridge records, the positive side follows
    the bad-triplet rules: one bridge link, one home-cluster triplet source,
    and extra full matches until ``match_bridge_ratio`` rows exist per link.
    Without bridge records, the generator samples positives from the declared
    full matches instead.

    :param id_table: Reference universe of the dataset. Records outside it are
        never sampled.
    :param matcher_gt: Pairwise ground truth. Non-zero labels supply positives;
        the structure of bridge records determines the composition.
    :param cluster_gt: Ideal clustering. When omitted, it is inferred from the
        matcher ground truth, which only succeeds for binary ground truths.
    :param excluded: Pairs that the comparison space must avoid. Pairs without
        a bridge record are hard exclusions; pairs with a bridge record avoid
        that bridge but bend under quota pressure.
    :param neg_pos_ratio: Number of non-match rows per positive row. Defaults
        to ``5.0``.
    :param match_bridge_ratio: Number of full-match rows per bridge link.
        Defaults to ``2.0``, one of which is the triplet source.
    :param max_total_samples: Upper bound on the number of rows. The cap can
        force fewer bridge links; it never forces more. Defaults to ``None``.
    :param seed: Seed for the ``numpy.random.default_rng`` stream behind link,
        match, and negative sampling. Defaults to ``42``.
    :param save: Write the comparison space to CSV when a path is given.
        Defaults to ``True``.
    :param log: Logger for generation diagnostics. Defaults to the package null
        logger.
    :param is_bridge: Predicate that marks bridge records. Defaults to ``None``,
        which derives the predicate from label-2 pairs. Binary ground truths
        must supply their own rule.
    :param allow_target_negatives: Let negatives join a bridge with members of
        its target clusters. Defaults to ``False``.
    """

    DIRECTED_CLASSES: ClassVar[list[int]] = [0, 1, 2, 3]
    UNDIRECTED_CLASSES: ClassVar[list[int]] = [0, 1]
    MIN_PER_CLASS: int = 3

    def __init__(
        self,
        id_table: IdTable,
        matcher_gt: dict[Pair, int],
        cluster_gt: frozenset[frozenset[RefId]] | None = None,
        excluded: Iterable[Pair] | None = None,
        neg_pos_ratio: float = 5.0,
        match_bridge_ratio: float = 2.0,
        max_total_samples: int | None = None,
        seed: int = 42,
        save: bool = True,
        log: logging.Logger | None = None,
        is_bridge: Callable[[RefId], bool] | None = None,
        allow_target_negatives: bool = False,
    ) -> None:
        self.neg_pos_ratio = neg_pos_ratio
        self.match_bridge_ratio = match_bridge_ratio
        self.max_total_samples = max_total_samples
        self._excluded = set(excluded) if excluded is not None else set()
        self._rng = np.random.default_rng(seed)
        self._log = resolve_logger(log)
        self._id_table = id_table
        self._matcher_gt = matcher_gt
        self._save = save
        self.is_bridge = is_bridge
        self.allow_target_negatives = allow_target_negatives
        self._cluster_gt = self.__init_cluster_gt(id_table, matcher_gt, cluster_gt)
        self._ref_id_cluster_map = self.__init_ref_id_cluster_numbers()
        self._cluster_members: dict[int, set[RefId]] = defaultdict(set)
        for ref, cluster in self._ref_id_cluster_map.items():
            self._cluster_members[cluster].add(ref)
        self._sampler: BadTripletSampler | None = None
        self._log.info(
            "%d entity references, %d clusters represented, %d unclustered",
            len(self._id_table),
            len(self._cluster_gt),
            sum(1 for ref in self._id_table if ref not in self._ref_id_cluster_map),
        )

    @property
    def seed(self) -> int | None:
        return getattr(self._rng.bit_generator.seed_seq, "entropy", None)

    @seed.setter
    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __init_ref_id_cluster_numbers(self) -> dict[RefId, int]:
        return {
            ref_id: cluster_number
            for cluster_number, cluster in enumerate(self._cluster_gt, 1)
            for ref_id in cluster
        }

    @staticmethod
    def __init_cluster_gt(
        id_table: IdTable,
        matcher_gt: dict[Pair, int],
        cluster_gt: frozenset[frozenset[RefId]] | None,
    ) -> frozenset[frozenset[RefId]]:
        if cluster_gt is not None:
            return cluster_gt
        n_classes = len(set(matcher_gt.values()))
        if n_classes > 2:
            raise ValueError(
                f"matcher ground truth with {n_classes} distinct classes: "
                "can't perform equivalence class partitioning"
            )
        ecp = EquivalenceClassPartitioner(id_table.ids())
        return ecp(matcher_gt)

    @property
    def true_clusters(self) -> frozenset[frozenset[RefId]]:
        return self._cluster_gt

    @property
    def true_matches(self) -> dict[Pair, int]:
        return self._matcher_gt

    def __call__(
        self,
        output_path: str | PathLike | None = None,
        clusters_output_path: str | PathLike | None = None,
    ) -> BinaryComparisonSpace:
        """Generate the comparison space and optionally save it as CSV.

        The generator chooses the composition from the bridge structure of the
        ground truth. With bridge records, link and full-match rows follow the
        bad-triplet rules; without them, the classic class-balanced sample is
        produced.

        :param output_path: File that receives the comparison space as CSV.
            Required when saving is enabled.
        :param clusters_output_path: Accepted for caller compatibility; the
            cluster assignment file is produced by the caller.
        :returns: The generated comparison space, in the same order as the CSV.
        :raises ValueError: If the ground truth has no usable positives, the
            bridge structure is malformed, a bridge is orphaned, or the sample
            cap is too small for the requested composition.
        """
        self._log.info(
            "Known matching pairs (non-zero class): %d", len(self._matcher_gt)
        )
        index = self._index()
        if index.bridges:
            comparisons = self._bridge_comparisons(index)
        else:
            comparisons = self._legacy_comparisons()
        comparison_space = InMemoryComparisonSpace()
        for left_id, right_id in comparisons:
            comparison_space.put(left_id, right_id)
        self._write_comparisons_csv(comparison_space, output_path)
        return comparison_space

    def _index(self) -> BridgeIndex:
        predicate = self.is_bridge or self._label_two_predicate()
        self._sampler = BadTripletSampler(
            self._id_table,
            self._matcher_gt,
            self._ref_id_cluster_map,
            predicate,
            log=self._log,
        )
        assert self._sampler is not None
        return self._sampler.index()

    def _label_two_predicate(self) -> Callable[[RefId], bool]:
        bridges = frozenset(
            right for (_, right), label in self._matcher_gt.items() if label == 2
        )
        return lambda ref: ref in bridges

    def _bridge_comparisons(self, index: BridgeIndex) -> list[Pair]:
        self._validate_bridge_index(index)
        assert self._sampler is not None
        sample = self._sampler.sample(exclusions=self._excluded)
        report = sample.exclusions
        if report.unknown_pairs:
            self._log.info(
                "ignored %d exclusion pairs absent from the ground truth",
                len(report.unknown_pairs),
            )
        self._log.info(
            "exclusions: %d hard pairs, %d avoided bridges, %d demoted bridges",
            len(report.hard_exclusions),
            len(report.avoided_bridges),
            len(report.demoted_bridges),
        )
        by_link = self._triplets_by_link(sample)
        selected_links = self._select_links(by_link, report)
        if not selected_links:
            raise ValueError("no bridge links available for the comparison space")
        selected_home_links = self._select_sources(selected_links, by_link)
        selected_home_extras = self._select_extras(
            index, report, selected_links, selected_home_links
        )
        positives = self._positive_pairs(
            selected_links, selected_home_links, selected_home_extras, index.bridges
        )
        negatives = self._bridge_negatives(index, len(positives))
        comparisons = positives + negatives
        self._rng.shuffle(comparisons)
        self._log.info(
            "comparison space: %d positives, %d negatives",
            len(positives),
            len(negatives),
        )
        return comparisons

    def _validate_bridge_index(self, index: BridgeIndex) -> None:
        fatal = [issue for issue in index.issues if issue[1] != "bridge_bridge"]
        if fatal:
            raise ValueError(f"invalid bridge structure: {fatal}")
        if ignored := [issue for issue in index.issues if issue[1] == "bridge_bridge"]:
            self._log.info("ignoring %d bridge-bridge pairs", len(ignored))
        if index.orphan_bridges:
            raise ValueError(f"orphan bridges: {index.orphan_bridges}")
        if not index.link_pairs:
            raise ValueError(
                "bridges are declared but no bridge links exist; binary ground "
                "truths must pass is_bridge"
            )

    @staticmethod
    def _triplets_by_link(sample: BadTripletSample) -> dict[Pair, list]:
        by_link = defaultdict(list)
        for triplet in sample.triplets:
            by_link[(triplet.target, triplet.bridge)].append(triplet)
        return by_link

    def _select_links(self, by_link, report) -> dict[RefId, list[Pair]]:
        target = self._link_target(len(by_link))
        strict = [link for link in by_link if link[1] not in report.avoided_bridges]
        bent = [link for link in by_link if link[1] in report.avoided_bridges]
        chosen = self._pick(strict, target)
        if len(chosen) < target:
            chosen.extend(self._pick(bent, target - len(chosen)))
        selected: dict[RefId, list[Pair]] = defaultdict(list)
        for link in chosen:
            selected[link[1]].append(link)
        return dict(selected)

    def _pick(self, pool: list[Pair], count: int) -> list[Pair]:
        if count <= 0:
            return []
        indices = self._rng.permutation(len(pool))[:count]
        return [pool[int(index)] for index in indices]

    def _link_target(self, available: int) -> int:
        if self.max_total_samples is None:
            return available
        cost = (1 + self.match_bridge_ratio) * (1 + self.neg_pos_ratio)
        limit = math.floor(self.max_total_samples / cost)
        if limit < 1:
            raise ValueError(
                "max_total_samples is too small for a comparison space that "
                "contains bridge links"
            )
        return min(available, limit)

    def _select_sources(
        self, selected: dict[RefId, list[Pair]], by_link
    ) -> dict[RefId, list[RefId]]:
        groups: dict[int, list[RefId]] = defaultdict(list)
        for bridge in selected:
            groups[self._ref_id_cluster_map[bridge]].append(bridge)
        sources: dict[RefId, list[RefId]] = {}
        for group in groups.values():
            sources.update(self._sample_sources(group, selected, by_link))
        return sources

    def _sample_sources(
        self, group: list[RefId], selected, by_link
    ) -> dict[RefId, list[RefId]]:
        pools = {}
        for bridge in group:
            homes = {}
            for link in selected[bridge]:
                for triplet in by_link[link]:
                    homes.setdefault(triplet.home, None)
            pool = list(homes)
            self._rng.shuffle(pool)
            pools[bridge] = pool
        remaining = {bridge: len(selected[bridge]) for bridge in group}
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

    def _select_extras(
        self, index: BridgeIndex, report, selected, sources
    ) -> dict[RefId, list[Pair]]:
        budget = {
            bridge: round(self.match_bridge_ratio * len(links)) - len(sources[bridge])
            for bridge, links in selected.items()
        }
        if missing := {bridge: n for bridge, n in budget.items() if n < 0}:
            raise ValueError(f"negative full-match budget {missing}")
        used = {
            frozenset((home, bridge))
            for bridge, homes in sources.items()
            for home in homes
        }
        for links in selected.values():
            used.update(frozenset(link) for link in links)
        extras: dict[RefId, list[Pair]] = defaultdict(list)
        bridge_free, home_pairs = self._full_match_pools(index, report)
        bridge_involving = [
            pair
            for bridge, pairs in home_pairs.items()
            if bridge not in report.avoided_bridges
            for pair in pairs
        ]
        avoided = [
            pair
            for bridge, pairs in home_pairs.items()
            if bridge in report.avoided_bridges
            for pair in pairs
        ]
        for pool in (bridge_free, bridge_involving, avoided):
            self._rng.shuffle(pool)
            self._assign_pairs(pool, budget, extras, used)
        if outstanding := {bridge: n for bridge, n in budget.items() if n > 0}:
            raise ValueError(f"match-bridge ratio not met for bridges {outstanding}")
        return extras

    def _full_match_pools(self, index: BridgeIndex, report):
        hard = {frozenset(pair) for pair in report.hard_exclusions}
        bridge_free: list[Pair] = []
        home_pairs: dict[RefId, list[Pair]] = defaultdict(list)
        seen: set[frozenset] = set()
        for pair, label in self._matcher_gt.items():
            if label != 1:
                continue
            key = frozenset(pair)
            if key in hard or key in seen:
                continue
            left, right = pair
            left_bridge = left in index.bridges
            right_bridge = right in index.bridges
            if left_bridge and right_bridge:
                continue
            seen.add(key)
            if not (left_bridge or right_bridge):
                bridge_free.append(pair)
                continue
            bridge = left if left_bridge else right
            home = right if left_bridge else left
            if self._ref_id_cluster_map.get(home) != self._ref_id_cluster_map.get(
                bridge
            ):
                continue
            home_pairs[bridge].append(pair)
        return bridge_free, dict(home_pairs)

    def _assign_pairs(self, candidates, budget, extras, used) -> None:
        index = 0
        while index < len(candidates) and any(need > 0 for need in budget.values()):
            order = [bridge for bridge, need in budget.items() if need > 0]
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
                budget[bridge] -= 1

    def _positive_pairs(self, selected, sources, extras, bridges) -> list[Pair]:
        pairs: list[Pair] = []
        for bridge, links in selected.items():
            pairs.extend((target, bridge) for target, _ in links)
            pairs.extend((home, bridge) for home in sources[bridge])
            pairs.extend(extras.get(bridge, []))
        return [self._orient(pair, bridges) for pair in pairs]

    @staticmethod
    def _orient(pair: Pair, bridges) -> Pair:
        left, right = pair
        if left in bridges and right not in bridges:
            return right, left
        return pair

    def _bridge_negatives(self, index: BridgeIndex, positives: int) -> list[Pair]:
        count = round(self.neg_pos_ratio * positives)
        if count <= 0:
            return []
        exclude = (
            set(self._matcher_gt) | set(self._excluded) | self._target_pairs(index)
        )
        generated = self._generate_negatives(count, exclude)
        pairs = [
            self._orient(pair, index.bridges)
            for pair in generated
            if not (pair[0] in index.bridges and pair[1] in index.bridges)
        ]
        if len(pairs) < count:
            self._log.warning("generated %d/%d negatives", len(pairs), count)
        if self.max_total_samples is not None:
            room = max(0, self.max_total_samples - positives)
            if len(pairs) > room:
                self._log.info(
                    "truncating negatives from %d to %d to respect the sample cap",
                    len(pairs),
                    room,
                )
                pairs = pairs[:room]
        return pairs

    def _target_pairs(self, index: BridgeIndex) -> set[Pair]:
        if self.allow_target_negatives:
            return set()
        pairs: set[Pair] = set()
        for bridge, link_pairs in index.link_pairs.items():
            for pair in link_pairs:
                target = pair[0] if pair[1] == bridge else pair[1]
                cluster = self._ref_id_cluster_map.get(target)
                if cluster is None:
                    continue
                pairs.update(
                    (other, bridge) for other in self._cluster_members[cluster]
                )
        return pairs

    def _legacy_comparisons(self) -> list[Pair]:
        by = self._legacy_by_class()
        pos_classes = sorted(by.keys() - {0})
        if not pos_classes:
            raise ValueError("no positive pairs available for the comparison space")
        pos_class_counts = {c: len(by[c]) for c in pos_classes}
        bridge_counts = [n for c, n in pos_class_counts.items() if c > 1]
        if bridge_counts:
            bridge_max = max([1, *bridge_counts])
            n1_target = self._clamp(
                round(self.match_bridge_ratio * bridge_max), pos_class_counts[1]
            )
        else:
            n1_target = pos_class_counts[1]
        total_pos = n1_target + sum(bridge_counts)
        n0_target = max(self.MIN_PER_CLASS, round(self.neg_pos_ratio * total_pos))
        raw_targets = {
            0: n0_target,
            1: n1_target,
            **{c: n for c, n in pos_class_counts.items() if c > 1},
        }
        targets = self._apply_cap(raw_targets)
        for class_no, target in raw_targets.items():
            if class_no == 0:
                self._log.info("target class 0: %d (to generate)", target)
            else:
                self._log.info(
                    "target class %d: %d (available %d)",
                    class_no,
                    targets.get(class_no, target),
                    pos_class_counts[class_no],
                )
        samples = {1: self._select_diverse(by.get(1, []), targets.get(1, 0))}
        for class_no, target in targets.items():
            if class_no < 2:
                continue
            pool = by.get(class_no, [])
            if target < len(pool):
                picked = self._rng.choice(len(pool), size=target, replace=False)
                samples[class_no] = [pool[int(i)] for i in picked]
            else:
                samples[class_no] = list(pool)
        exclude = self._excluded.union(set(self._matcher_gt))
        samples[0] = self._generate_negatives(targets.get(0, 0), exclude)
        comparisons = [pair for items in samples.values() for pair in items]
        self._rng.shuffle(comparisons)
        return comparisons

    def _legacy_by_class(self) -> dict[int, list[Pair]]:
        excluded = {frozenset(pair) for pair in self._excluded}
        by: dict[int, list[Pair]] = defaultdict(list)
        for pair, class_no in self._matcher_gt.items():
            if not class_no or frozenset(pair) in excluded:
                continue
            by[class_no].append(pair)
        return dict(by)

    def _apply_cap(self, targets: dict[int, int]) -> dict[int, int]:
        if self.max_total_samples is None:
            return dict(targets)

        classes = list(targets)
        n_classes = len(classes)
        total = sum(targets.values())
        if total <= self.max_total_samples:
            self._log.info(
                "cap check: %d ≤ %d; no scaling needed", total, self.max_total_samples
            )
            return dict(targets)

        self._log.info(
            "cap check: %d > %d; scaling down proportionally",
            total,
            self.max_total_samples,
        )

        budget = self.max_total_samples
        reserved = self.MIN_PER_CLASS * n_classes

        if budget < reserved:
            raise ValueError(
                f"max_total_samples ({budget}) is too small to guarantee "
                f"≥{self.MIN_PER_CLASS} per class × {n_classes} classes = "
                f"{reserved} minimum."
            )

        remaining = budget - reserved
        proportional_total = sum(
            max(targets[c] - self.MIN_PER_CLASS, 0) for c in classes
        )

        result: dict[int, int] = {}
        allocated = 0
        for class_no in classes:
            excess = max(targets[class_no] - self.MIN_PER_CLASS, 0)
            if proportional_total > 0:
                share = math.floor(remaining * excess / proportional_total)
            else:
                share = math.floor(remaining / n_classes)
            result[class_no] = self.MIN_PER_CLASS + share
            allocated += result[class_no]

        leftover = budget - allocated
        if leftover > 0:
            largest = max(classes, key=lambda c: result[c])
            result[largest] += leftover

        self._log.info(
            "scaled targets: %s (total %d)",
            ", ".join(f"{k}={v}" for k, v in result.items()),
            sum(result.values()),
        )
        return result

    def _generate_negatives(self, n: int, exclude: set[Pair]) -> list[Pair]:
        self._log.info("Generating %d negatives", n)

        pool = [ref.id for ref in self._id_table]
        n_pool = len(pool)
        if n_pool < 2:
            raise ValueError("Record pool has < 2 entries; cannot generate pairs.")

        seen: set[Pair] = set()
        attempts = 0
        max_attempts = n * 100

        result: list[Pair] = []
        while len(result) < n and attempts < max_attempts:
            batch = min((n - len(result)) * 5, 100_000)
            i1 = self._rng.integers(0, n_pool, size=batch)
            i2 = self._rng.integers(0, n_pool, size=batch)
            attempts += batch

            for k in range(batch):
                if len(result) >= n:
                    break

                left_ref_id = pool[int(i1[k])]
                right_ref_id = pool[int(i2[k])]
                if left_ref_id == right_ref_id:
                    continue

                cl_l = self._ref_id_cluster_map.get(left_ref_id)
                cl_r = self._ref_id_cluster_map.get(right_ref_id)
                if cl_l is not None and cl_r is not None and cl_l == cl_r:
                    continue

                fwd = (left_ref_id, right_ref_id)
                rev = (right_ref_id, left_ref_id)
                if fwd in exclude or rev in exclude:
                    continue
                if fwd in seen or rev in seen:
                    continue

                seen.add(fwd)
                result.append(fwd)

        if len(result) < n:
            self._log.warning(
                "Only %d/%d negatives after %d attempts", len(result), n, attempts
            )
        else:
            self._log.info("Got %d negatives after %d attempts", n, attempts)

        return result

    def _select_diverse(self, pool: list[Pair], n: int) -> list[Pair]:
        if n >= len(pool):
            return list(pool)

        freq: Counter = Counter()
        for left_id, right_id in pool:
            freq[left_id] += 1
            freq[right_id] += 1

        scores = np.array(
            [freq[left_id] + freq[right_id] for left_id, right_id in pool],
            dtype=np.float64,
        )
        scores += self._rng.random(len(scores)) * 0.01

        idx = np.argsort(scores)[:n]
        result = [pool[int(index)] for index in idx]
        unique = {ref for pair in result for ref in pair}
        self._log.info(
            "diversity: kept %d/%d, %d unique IDs", n, len(pool), len(unique)
        )
        return result

    @staticmethod
    def _clamp(target: int, available: int) -> int:
        return max(
            GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS,
            min(target, available),
        )

    def _label_of(self, pair: Pair) -> int:
        if pair in self._matcher_gt:
            return self._matcher_gt[pair]
        return self._matcher_gt.get((pair[1], pair[0]), 0)

    def _write_comparisons_csv(
        self,
        comparison_space: InMemoryComparisonSpace,
        output_path: str | PathLike | None,
    ) -> pl.DataFrame:
        if not self._save:
            return pl.DataFrame()

        comparison_data = []
        for left_id, right_id in comparison_space:
            comparison_data.append(
                {
                    "left_id": left_id.label,
                    "left_source": left_id.source,
                    "right_id": right_id.label,
                    "right_source": right_id.source,
                    "label": self._label_of((left_id, right_id)),
                }
            )
        comparisons_df = pl.DataFrame(comparison_data)
        self._log.info(
            "saving comparison space with match labels: %d comparisons, %s",
            len(comparisons_df),
            ", ".join(
                f"{c}={n}"
                for c, n in comparisons_df["label"].value_counts(sort=True).iter_rows()
            ),
        )
        comparisons_df.write_csv(output_path, include_header=True)
        return comparisons_df
