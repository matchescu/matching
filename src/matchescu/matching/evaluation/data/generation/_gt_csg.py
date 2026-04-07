import logging
import math
import random
from os import PathLike
from typing import Optional, Counter, Iterable

import numpy as np
import polars as pl

from matchescu.matching.evaluation.ground_truth import EquivalenceClassPartitioner
from matchescu.reference_store.comparison_space import (
    BinaryComparisonSpace,
    InMemoryComparisonSpace,
)
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId


class GroundTruthComparisonSpaceGenerator(object):
    DIRECTED_CLASSES: list[int] = [0, 1, 2, 3]
    UNDIRECTED_CLASSES: list[int] = [0, 1]
    MIN_PER_CLASS: int = 3

    def __init__(
        self,
        id_table: IdTable,
        matcher_gt: dict[tuple[RefId, RefId], int],
        cluster_gt: frozenset[frozenset[RefId]] | None = None,
        excluded: Iterable[tuple[RefId, RefId]] | None = None,
        neg_pos_ratio: float = 5.0,
        match_bridge_ratio: float = 2.0,
        max_total_samples: Optional[int] = None,
        seed: int = 42,
        save_comparisons: bool = True,
        save_clusters: bool = True,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize `GroundTruthComparisonSpaceGenerator`.

        Args:
            id_table:
                table mapping entity reference identifiers to entity references
            matcher_gt:
                ground truth for pairwise matching (contains true matches)
            cluster_gt:
                ground truth for clustering (provides the ideal clustering
                expected for the supplied ``id_table``)
            neg_pos_ratio:
                ``#non-match / #match + #potential_match``.  Default ``5.0``.
            match_bridge_ratio:
                ``#class-1 / max(#class-2, #class-3, ..., #class-n)`` ceiling.  Default ``2.0``.
            max_total_samples:
                Hard upper limit on total rows across all splits (train+dev+test).
                When set, all class counts are scaled down proportionally while
                preserving the configured ratios as closely as possible.
                ``None`` (default) means no limit.
            seed:
                Seed for reproducibility.
        """
        self.neg_pos_ratio = neg_pos_ratio
        self.match_bridge_ratio = match_bridge_ratio
        self.max_total_samples = max_total_samples
        self.random_state = seed
        self._excluded = set(excluded) if excluded is not None else set()
        self._rng = np.random.RandomState(seed)
        self._log = log or logging.getLogger(self.__class__.__name__)
        self._id_table = id_table
        self._matcher_gt = matcher_gt
        self._save_comparisons = save_comparisons
        self._cluster_gt = self.__init_cluster_gt(id_table, matcher_gt, cluster_gt)
        self._save_clusters = save_clusters
        self._ref_id_cluster_map = {
            ref_id: cluster_no
            for cluster_no, cluster in enumerate(self._cluster_gt, start=1)
            for ref_id in cluster
        }
        self._log.info(
            "%d entity references, %d clusters represented, %d unclustered",
            len(self._id_table),
            len(self._cluster_gt),
            sum(1 for ref in self._id_table if ref not in self._ref_id_cluster_map),
        )

    @staticmethod
    def __init_cluster_gt(
        id_table: IdTable,
        matcher_gt: dict[tuple[RefId, RefId], int],
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
    def true_matches(self) -> dict[tuple[RefId, RefId], int]:
        return self._matcher_gt

    def __call__(
        self,
        output_path: str | PathLike | None = None,
        clusters_output_path: str | PathLike | None = None,
    ) -> BinaryComparisonSpace:
        """Generate negatives, balance classes and optionally save output as CSV.

        If the ``output_path`` is not given, but saving the output is enabled a
        ``ValueError`` is raised. If ``clusters_output_path`` is not given, but
        saving clusters is enabled, a ``ValueError`` is raised.

        :param output_path: path to file where we save the generated comparison
            space, along with match ground truth labels as CSV
        :param clusters_output_path: path to file where cluster assignments are
            saved for each reference identifier.
        """
        self._log.info(
            "Known matching pairs (non-zero class): %d", len(self._matcher_gt)
        )
        pos_classes = list(sorted(set(self._matcher_gt.values()) - {0}))
        by: dict[int, list[tuple]] = {}
        for cmp, c in self._matcher_gt.items():
            by.setdefault(c, []).append(cmp)

        pos_class_counts = {c: len(by[c]) for c in pos_classes}
        self._log.info(
            "matcher ground truth positive classes: %s",
            ", ".join(f"[{c}] = {n}" for c, n in pos_class_counts.items()),
        )
        bridge_class_counts = [n for c, n in pos_class_counts.items() if c > 1]
        if len(bridge_class_counts) > 0:
            bridge_max = max([1, *bridge_class_counts])
            n1_target = self._clamp(
                int(round(self.match_bridge_ratio * bridge_max)), pos_class_counts[1]
            )
        else:
            n1_target = pos_class_counts[1]

        total_pos = n1_target + sum(bridge_class_counts)
        n0_target = max(self.MIN_PER_CLASS, int(round(self.neg_pos_ratio * total_pos)))

        # ── apply max_total_samples cap ──
        raw_targets = {
            0: n0_target,
            1: n1_target,
            **{c: n for c, n in pos_class_counts.items() if c > 1},
        }
        targets = self._apply_cap(raw_targets)
        for c, n in raw_targets.items():
            match c:
                case 0:
                    self._log.info("target class 0: %d (to generate)", n)
                case 1:
                    self._log.info("target class 1: %d (from %d)", n, raw_targets[1])
                case _:
                    self._log.info(
                        "target class %d: %d (available %d)", c, n, raw_targets[c]
                    )
        samples = {1: self._select_diverse(by[1], targets[1])}
        for c, n in targets.items():
            if c < 2:
                continue
            class_pool = random.sample(by[c], k=n) if n < raw_targets[c] else by[c]
            samples.update({c: class_pool})
        exclude = self._excluded.union(set(self._matcher_gt))
        samples.update({0: self._generate_negatives(targets[0], exclude)})

        comparison_space = InMemoryComparisonSpace()
        for label, items in samples.items():
            for left_id, right_id in items:
                comparison_space.put(left_id, right_id)

        self._write_comparisons_csv(comparison_space, output_path)
        self._write_clusters_csv(comparison_space, clusters_output_path)

        return comparison_space

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

        # Reserve minimum slots for every class first
        budget = self.max_total_samples
        reserved = self.MIN_PER_CLASS * n_classes

        if budget < reserved:
            raise ValueError(
                f"max_total_samples ({budget}) is too small to guarantee "
                f"≥{self.MIN_PER_CLASS} per class × {n_classes} classes = "
                f"{reserved} minimum."
            )

        # Distribute the remaining budget proportionally
        remaining = budget - reserved
        proportional_total = sum(
            max(targets[c] - self.MIN_PER_CLASS, 0) for c in classes
        )

        result: dict[int, int] = {}
        allocated = 0
        for c in classes:
            excess = max(targets[c] - self.MIN_PER_CLASS, 0)
            if proportional_total > 0:
                share = math.floor(remaining * excess / proportional_total)
            else:
                share = math.floor(remaining / n_classes)
            result[c] = self.MIN_PER_CLASS + share
            allocated += result[c]

        # Distribute any leftover from rounding to the largest class
        leftover = budget - allocated
        if leftover > 0:
            largest = max(classes, key=lambda c: result[c])
            result[largest] += leftover

        self._log.info(
            "scaled targets: %s (total %d))",
            ", ".join(f"{k}={v}" for k, v in result.items()),
            sum(result.values()),
        )
        return result

    def _generate_negatives(
        self, n: int, exclude: set[tuple[RefId, RefId]]
    ) -> list[tuple[RefId, RefId]]:
        self._log.info("Generating %d negatives", n)

        pool = [ref.id for ref in self._id_table]
        n_pool = len(pool)
        if n_pool < 2:
            raise ValueError("Record pool has < 2 entries; cannot generate pairs.")

        seen: set[tuple] = set()
        attempts = 0
        max_attempts = n * 100

        result: list[tuple[RefId, RefId]] = []
        while len(result) < n and attempts < max_attempts:
            batch = min((n - len(result)) * 5, 100_000)
            i1 = self._rng.randint(0, n_pool, size=batch)
            i2 = self._rng.randint(0, n_pool, size=batch)
            attempts += batch

            for k in range(batch):
                if len(result) >= n:
                    break

                left_ref_id = pool[i1[k]]
                right_ref_id = pool[i2[k]]
                if left_ref_id == right_ref_id:
                    continue

                # we have to rely on the cluster ground truth for multi-class matchers
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

    def _select_diverse(self, pool: list[tuple], n: int) -> list[tuple]:
        if n >= len(pool):
            return pool.copy()

        freq: Counter = Counter()
        for left_id, right_id in pool:
            if (left_id, right_id) in self._excluded:
                freq[left_id] = -1
                freq[right_id] = -1
            else:
                freq[left_id] += 1
                freq[right_id] += 1

        scores = np.array(
            [freq[left_id] + freq[right_id] for left_id, right_id in pool],
            dtype=np.float64,
        )
        scores += self._rng.random(len(scores)) * 0.01

        idx = np.argsort(scores)[:n]
        result = np.array(pool)[idx].tolist()
        uq = set()
        for x, y in result:
            uq.add(x)
            uq.add(y)
        n_uniq = len(uq)
        self._log.info("diversity: kept %d/%d, %d unique IDs", n, len(pool), n_uniq)

        return result

    @staticmethod
    def _clamp(target: int, available: int) -> int:
        return max(
            GroundTruthComparisonSpaceGenerator.MIN_PER_CLASS, min(target, available)
        )

    def _write_clusters_csv(
        self,
        comparison_space: InMemoryComparisonSpace,
        clusters_output_path: str | PathLike | None,
    ) -> pl.DataFrame:
        if not self._save_clusters:
            return pl.DataFrame()

        # handle clusters
        clusters: dict[int, set[RefId]] = {}
        for tpl in comparison_space:
            if tpl not in self.true_matches:
                continue
            for ref_id in tpl:
                if (cluster_no := self._ref_id_cluster_map.get(ref_id)) is None:
                    continue
                clusters.setdefault(cluster_no, set()).add(ref_id)
        cluster_data = []
        for (
            cluster_no,
            cluster,
        ) in enumerate(  # renumber the clusters (maybe not such a good idea?)
            (c for c in clusters.values() if len(c) > 1), start=1  # skip singletons
        ):
            for ref_id in cluster:
                cluster_data.append(
                    {
                        "id": ref_id.label,
                        "source": ref_id.source,
                        "cluster_id": cluster_no,
                    }
                )
        clusters_df = pl.DataFrame(cluster_data).sort(by="cluster_id")
        self._log.info(
            "saving clusters: %d clusters", clusters_df.n_unique(subset=["cluster_id"])
        )
        clusters_df.write_csv(clusters_output_path, include_header=True)
        return clusters_df

    def _write_comparisons_csv(
        self,
        comparison_space: InMemoryComparisonSpace,
        output_path: str | PathLike | None,
    ) -> pl.DataFrame:
        if not self._save_comparisons:
            return pl.DataFrame()

        # handle comparisons and match labels
        comparison_data = []
        for tpl in comparison_space:
            left_id, right_id = tpl
            comparison_data.append(
                {
                    "left_id": left_id.label,
                    "left_source": left_id.source,
                    "right_id": right_id.label,
                    "right_source": right_id.source,
                    "label": self._matcher_gt.get(tpl, 0),
                }
            )
        comparisons_df = pl.DataFrame(comparison_data).sample(fraction=1, shuffle=True)
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
