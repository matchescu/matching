import argparse
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
from matchescu.extraction import RecordExtraction, Traits, single_record
from matchescu.extraction.csv import CsvFile
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.reference_store.id_table import IdTable, InMemoryIdTable
from matchescu.typing import EntityReference, EntityReferenceIdentifier

from matchescu.matching._logging import resolve_logger
from matchescu.matching.evaluation.data.splits._split import Split

DIRECTED_CLASSES: list[int] = [0, 1, 2, 3]
UNDIRECTED_CLASSES: list[int] = [0, 1]
MIN_PER_CLASS: int = 3


def _pairs_frame(samples: dict[int, list[tuple]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "left_id": left_id.label,
            "left_source": left_id.source,
            "right_id": right_id.label,
            "right_source": right_id.source,
            "label": c,
        }
        for c, items in samples.items()
        for left_id, right_id in items
    )


def _class_counts(frame: pl.DataFrame) -> str:
    return ", ".join(
        f"{c}={n}" for c, n in frame["label"].value_counts(sort=True).iter_rows()
    )


class SplitGenerator:
    """Generates balanced train / dev / test splits.

    The input must provide the following:

    * an ``IdTable`` - containing a mapping from ID to entity reference for all
      entity references in the input domain of the ER task
    * a ground truth containing the ideal clustering (without singletons) of the
      entire ID table
    * a ground truth containing the pairs of matches
    """

    def __init__(
        self,
        split_ratio: dict[str, int] | None = None,
        neg_pos_ratio: float = 5.0,
        match_bridge_ratio: float = 2.0,
        max_total_samples: int | None = None,
        seed: int = 42,
        log: logging.Logger | None = None,
    ) -> None:
        """Initialize `SplitGenerator`.

        Args:
            split_ratio:
            ``(train, dev, test)`` integer ratio.  Default ``(3, 1, 1)``.
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
        self.split_ratio = split_ratio or {"train": 3, "dev": 1, "test": 1}
        self.neg_pos_ratio = neg_pos_ratio
        self.match_bridge_ratio = match_bridge_ratio
        self.max_total_samples = max_total_samples
        self.random_state = seed
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._log = resolve_logger(log)

        self._id_table: IdTable | None = None
        self._matcher_gt: dict[
            tuple[EntityReferenceIdentifier, EntityReferenceIdentifier], int
        ] = {}
        self._cluster_gt: dict[EntityReferenceIdentifier, int] = {}
        self._clusters: dict[int, set[EntityReferenceIdentifier]] = defaultdict(set)
        self._record_pool: list[EntityReference] = []
        self._splits: dict[str, pl.DataFrame] = {}

    def load(
        self,
        id_table: IdTable,
        cluster_gt: dict[EntityReferenceIdentifier, int],
        matcher_gt: dict[
            tuple[EntityReferenceIdentifier, EntityReferenceIdentifier], int
        ],
    ) -> "SplitGenerator":
        self._id_table = id_table
        self._cluster_gt = cluster_gt
        self._matcher_gt = matcher_gt
        self._clusters.clear()
        for ref_id, cluster_no in cluster_gt.items():
            self._clusters[cluster_no].add(ref_id)

        unclustered = 0
        for ref in self._id_table:
            if ref.id not in self._cluster_gt:
                unclustered += 1

        self._log.info(
            "%d entity references, %d clusters represented, %d unclustered",
            len(self._id_table),
            len(self._clusters),
            unclustered,
        )
        return self

    def generate(self) -> "SplitGenerator":
        """Generate negatives, balance classes, split into train/dev/test."""
        self._check_loaded()

        by = self._pairs_by_class()
        pos_classes = sorted(by.keys() - {0})
        targets = self._plan_targets(by, pos_classes)
        samples = {1: self._select_diverse(by[1], targets[1])}
        for c, n in targets.items():
            if c < 2:
                continue
            rand_idx = self._rng.choice(len(by[c]), size=n, replace=False)
            samples.update({c: [by[c][i] for i in rand_idx]})
        samples.update({0: self._generate_negatives(targets[0], set(self._matcher_gt))})

        d_bal = _pairs_frame(samples)
        self._log.info(
            "balanced: %d, %s",
            len(d_bal),
            _class_counts(d_bal),
        )
        self._splits = self._stratified_split(d_bal, [0, *pos_classes])
        return self

    def _pairs_by_class(self) -> dict[int, list[tuple]]:
        by: dict[int, list[tuple]] = defaultdict(list)
        for cmp, c in self._matcher_gt.items():
            by[c].append(cmp)
        self._log.info(
            "Known matching pairs (non-zero class): %d", len(self._matcher_gt)
        )
        self._log.info(
            "matcher ground truth positive classes: %s",
            ", ".join(f"[{c}] = {len(by[c])}" for c in sorted(set(by) - {0})),
        )
        return by

    def _plan_targets(
        self, by: dict[int, list[tuple]], pos_classes: list[int]
    ) -> dict[int, int]:
        pos_class_counts = {c: len(by[c]) for c in pos_classes}
        bridge_classes = [c for c in pos_classes if c > 1]
        avail_bridge_counts = {c: pos_class_counts[c] for c in bridge_classes}
        avail_bridge_total = sum(avail_bridge_counts.values())
        avail_n1 = pos_class_counts.get(1, 0)

        bridge_targets, n1_target = self._bridge_targets(
            bridge_classes, avail_bridge_counts, avail_bridge_total, avail_n1
        )
        total_pos = n1_target + sum(bridge_targets.values())
        n0_target = max(MIN_PER_CLASS, round(self.neg_pos_ratio * total_pos))
        raw_targets = {0: n0_target, 1: n1_target, **bridge_targets}
        targets = self._apply_cap(raw_targets)
        self._log_targets(raw_targets)
        return targets

    def _bridge_targets(
        self,
        bridge_classes: list[int],
        avail_bridge_counts: dict[int, int],
        avail_bridge_total: int,
        avail_n1: int,
    ) -> tuple[dict[int, int], int]:
        if avail_bridge_total == 0 or self.match_bridge_ratio <= 0:
            return avail_bridge_counts.copy(), avail_n1
        needed_n1 = round(self.match_bridge_ratio * avail_bridge_total)
        if avail_n1 >= needed_n1:
            return avail_bridge_counts.copy(), self._clamp(needed_n1, avail_n1)
        target_bridge_total = max(
            MIN_PER_CLASS * len(bridge_classes),
            round(avail_n1 / self.match_bridge_ratio),
        )
        target_bridge_total = min(target_bridge_total, avail_bridge_total)
        bridge_targets = self._allocate_bridge_targets(
            bridge_classes, avail_bridge_counts, avail_bridge_total, target_bridge_total
        )
        actual_bridge_total = sum(bridge_targets.values())
        n1_target = round(self.match_bridge_ratio * actual_bridge_total)
        return bridge_targets, self._clamp(n1_target, avail_n1)

    def _allocate_bridge_targets(
        self,
        bridge_classes: list[int],
        avail_bridge_counts: dict[int, int],
        avail_bridge_total: int,
        target_bridge_total: int,
    ) -> dict[int, int]:
        bridge_targets = {}
        allocated = 0
        for c in bridge_classes:
            available = avail_bridge_counts[c]
            share = round(target_bridge_total * (available / avail_bridge_total))
            count = self._clamp(share, available)
            bridge_targets[c] = count
            allocated += count
        discrepancy = target_bridge_total - allocated
        if discrepancy != 0 and bridge_classes:
            largest_bridge = max(bridge_classes, key=lambda k: bridge_targets[k])
            bridge_targets[largest_bridge] = self._clamp(
                bridge_targets[largest_bridge] + discrepancy,
                avail_bridge_counts[largest_bridge],
            )
        return bridge_targets

    def _log_targets(self, raw_targets: dict[int, int]) -> None:
        for c, n in raw_targets.items():
            match c:
                case 0:
                    self._log.info("target class 0: %d (to generate)", n)
                case 1:
                    self._log.info("target class 1: %d", n)
                case _:
                    self._log.info("target class %d: %d", c, n)

    def get_splits(self) -> dict[str, Split]:
        if self._splits is None or len(self._splits) == 0:
            raise RuntimeError("generate() before returning splits")

        splits = {}
        for split_name, df in self._splits.items():
            split_pairs = [
                (
                    EntityReferenceIdentifier(row["left_id"], row["left_source"]),
                    EntityReferenceIdentifier(row["right_id"], row["right_source"]),
                )
                for row in df.iter_rows(named=True)
            ]
            cs = InMemoryComparisonSpace()
            match_gt = {}
            cluster_gt = {}
            for x, y in split_pairs:
                cs.put(x, y)
                fwd_cmp = (x, y)
                rev_cmp = (y, x)
                label = self._matcher_gt.get(fwd_cmp, 0)
                if label == 0 and rev_cmp in self._matcher_gt:
                    self._log.warning(
                        "pair (%s, %s) appears in reverse order; treating it as a non-match",
                        x,
                        y,
                    )
                if label != 0:
                    match_gt[fwd_cmp] = label

                    x_cluster = self._cluster_gt.get(x)
                    y_cluster = self._cluster_gt.get(y)
                    if x_cluster is not None:
                        cluster_gt[x] = x_cluster
                    if y_cluster is not None:
                        cluster_gt[y] = y_cluster

            clusters = defaultdict(set)
            for ref_id, cluster_no in cluster_gt.items():
                clusters[cluster_no].add(ref_id)
            splits[split_name] = Split(cs, match_gt, clusters)
        return splits

    def save(
        self,
        output_dir: str | os.PathLike,
        prefix: str = "",
        file_names: Mapping[str, str] | None = None,
    ) -> "SplitGenerator":
        """Write the split CSV files to *output_dir*."""
        if self._splits is None:
            raise RuntimeError(
                "call generate(): attempting to save without generating splits"
            )

        output = Path(output_dir)
        self._log.info("Saving splits to: %s", output)
        for split_name, df in self._splits.items():
            path = output / self._file_name(split_name, prefix, file_names)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(path, include_header=True)
            self._log.info("Saved %d rows to file %s", len(df), path.name)
        return self

    @staticmethod
    def _file_name(
        split_name: str,
        prefix: str,
        file_names: Mapping[str, str] | None,
    ) -> str:
        if file_names is not None and split_name in file_names:
            return file_names[split_name]
        return f"{prefix}_{split_name}.csv" if prefix else f"{split_name}.csv"

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
        reserved = MIN_PER_CLASS * n_classes

        if budget < reserved:
            raise ValueError(
                f"max_total_samples ({budget}) is too small to guarantee "
                f"≥{MIN_PER_CLASS} per class × {n_classes} classes = "
                f"{reserved} minimum."
            )

        # Distribute the remaining budget proportionally
        remaining = budget - reserved
        proportional_total = sum(max(targets[c] - MIN_PER_CLASS, 0) for c in classes)

        result: dict[int, int] = {}
        allocated = 0
        for c in classes:
            excess = max(targets[c] - MIN_PER_CLASS, 0)
            if proportional_total > 0:
                share = math.floor(remaining * excess / proportional_total)
            else:
                share = math.floor(remaining / n_classes)
            result[c] = MIN_PER_CLASS + share
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
        self,
        n: int,
        exclude: set[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]],
        pool: list[EntityReferenceIdentifier] | None = None,
    ) -> list[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]]:
        """Sample *n_needed* cross-cluster negative pairs.

        A candidate pair is accepted iff:
        1. Not the same record.
        2. Different clusters.
        3. Not a known positive.
        4. Not already generated (no duplicates).
        """
        self._log.info("Generating %d negatives", n)

        if pool is None:
            pool = [ref.id for ref in self._id_table]
        n_pool = len(pool)
        if n_pool < 2:
            raise ValueError("Record pool has < 2 entries; cannot generate pairs.")

        seen: set[tuple] = set()
        attempts = 0
        max_attempts = n * 100

        result: list[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]] = []
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

                cl_l = self._cluster_gt.get(left_ref_id)
                cl_r = self._cluster_gt.get(right_ref_id)
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
        """Pick *n* pairs preferring ID diversity (fewer hub-dominated
        star formations)."""
        if n >= len(pool):
            return pool.copy()

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
        if available < MIN_PER_CLASS:
            raise ValueError(
                f"class has {available} pairs, fewer than MIN_PER_CLASS={MIN_PER_CLASS}"
            )
        return max(MIN_PER_CLASS, min(target, available))

    def _stratified_split(
        self, df: pl.DataFrame, classes: list[int]
    ) -> dict[str, pl.DataFrame]:
        """Per-class split respecting *split_ratio*, >=1 per class per split."""
        parts: dict[str, list[pl.DataFrame]] = {k: [] for k in self.split_ratio}
        for c in classes:
            sub = self._shuffled_class(df, c)
            counts = self._split_counts(len(sub), list(self.split_ratio.values()))
            slices = self._class_slices(sub, counts)
            for name, frame in zip(parts, slices):
                parts[name].append(frame)
        return {
            name: self._shuffle_frame(pl.concat(frames, how="vertical", strict=True))
            for name, frames in parts.items()
        }

    def _shuffled_class(self, df: pl.DataFrame, label: int) -> pl.DataFrame:
        sub = df.filter(pl.col("label") == label)
        assert len(sub) >= MIN_PER_CLASS, f"class {label}: {len(sub)} < {MIN_PER_CLASS}"
        seed = int(self._rng.randint(0, np.iinfo(np.int32).max))
        return sub.sample(fraction=1, seed=seed, shuffle=True, with_replacement=False)

    @staticmethod
    def _split_counts(n: int, ratios: list[int]) -> list[int]:
        total = sum(ratios)
        counts = [max(1, round((n * r) / total)) for r in ratios[1:]]
        n_train = n - sum(counts)
        while n_train < 1 and any(count > 1 for count in counts):
            index = max(range(len(counts)), key=lambda i: counts[i])
            counts[index] -= 1
            n_train = n - sum(counts)
        return [n_train, *counts]

    @staticmethod
    def _class_slices(sub: pl.DataFrame, counts: list[int]) -> list[pl.DataFrame]:
        slices = []
        start = 0
        for count in counts:
            slices.append(sub[start : start + count])
            start += count
        return slices

    def _shuffle_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sample(
            fraction=1,
            seed=self.random_state,
            shuffle=True,
            with_replacement=False,
        )

    def _check_loaded(self) -> None:
        if any(x is None for x in (self._id_table, self._cluster_gt, self._matcher_gt)):
            raise RuntimeError("Call load() before generate().")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate balanced train/dev/test splits for "
        "affiliation-string matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Negative pairs (label 0) are GENERATED from cross-cluster ID pairs.
Ground-truth files may contain as many non-zero labels as desired. The number of
classes differs for directed vs undirected matcher ground truths.

Directed matcher ground truths: 
    0 = no match (generated)
    1 = two-way match   2 = bridge L → R   3 = bridge R → L

Undirected matcher ground truths: 
    0 = no match (generated)
    1 = match

Output (in --output-dir): train.csv, test.csv, dev.csv, optionally prefixed. 
""",
    )
    ap.add_argument(
        "--matcher-ground-truth",
        required=True,
        help="CSV file containing the matcher ground truth",
    )
    ap.add_argument(
        "--cluster-ground-truth",
        required=True,
        help="ground truth containing cluster assignment per entity reference",
    )
    ap.add_argument(
        "--data-source",
        required=True,
        help="CSV file containing data to be extracted",
    )
    ap.add_argument("--output-dir", default="splits")
    ap.add_argument(
        "--split-ratio",
        nargs=3,
        type=int,
        default=[3, 1, 1],
        metavar=("TR", "DV", "TS"),
        help="train : dev : test  (default: 3 1 1)",
    )
    ap.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=8.0,
        help="neg / pos ratio  (default: 8; recommended 3–10)",
    )
    ap.add_argument(
        "--match-bridge-ratio",
        type=float,
        default=3.0,
        help="class-1 / bridge-class ceiling  (default: 2; recommended <5)",
    )
    ap.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="hard upper limit on total rows (train+dev+test)." "Default: no limit.",
    )
    ap.add_argument("--random-state", type=int, default=42, help="random seed")
    a = ap.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    traits = list(Traits().string(["affil1"]))
    ds = CsvFile(a.data_source, traits, has_header=True).read()

    def _ref_id(records):
        rec = next(iter(records))
        return EntityReferenceIdentifier(rec["id1"], rec["source"])

    extract = RecordExtraction(ds, _ref_id, single_record)
    id_table = InMemoryIdTable()
    for x in extract():
        id_table.put(x)
    match_gt_df = pl.read_csv(a.matcher_ground_truth)
    match_gt = {
        (
            EntityReferenceIdentifier(x["left_id"], x["left_source"]),
            EntityReferenceIdentifier(x["right_id"], x["right_source"]),
        ): x["label"]
        for x in match_gt_df.iter_rows(named=True)
    }
    cluster_gt_df = pl.read_csv(a.cluster_ground_truth)
    cluster_gt = {
        EntityReferenceIdentifier(x["id"], x["source"]): x["cluster_id"]
        for x in cluster_gt_df.iter_rows(named=True)
    }
    split_ratios = dict(zip(["train", "dev", "test"], a.split_ratio))
    generator = (
        SplitGenerator(
            split_ratio=split_ratios,
            neg_pos_ratio=a.neg_pos_ratio,
            match_bridge_ratio=a.match_bridge_ratio,
            max_total_samples=a.max_total_samples,
            seed=a.random_state,
            log=logging.getLogger(__name__),
        )
        .load(id_table, cluster_gt, match_gt)
        .generate()
    )
    generator.save(a.output_dir)


if __name__ == "__main__":
    main()
