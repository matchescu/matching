import argparse
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import polars as pl

from matchescu.data_sources import CsvDataSource
from matchescu.extraction import Traits, single_record, RecordExtraction
from matchescu.matching.evaluation.data.splits._split import Split
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.reference_store.id_table import IdTable, InMemoryIdTable
from matchescu.typing import EntityReferenceIdentifier, EntityReference

DIRECTED_CLASSES: list[int] = [0, 1, 2, 3]
UNDIRECTED_CLASSES: list[int] = [0, 1]
MIN_PER_CLASS: int = 3


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
        split_ratio: dict[str, int] = None,
        neg_pos_ratio: float = 5.0,
        match_bridge_ratio: float = 2.0,
        max_total_samples: Optional[int] = None,
        seed: int = 42,
        log: Optional[logging.Logger] = None,
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
        self._rng = np.random.RandomState(seed)
        self._log = log or logging.getLogger(self.__class__.__name__)

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

        self._log.info(
            "Known matching pairs (non-zero class): %d", len(self._matcher_gt)
        )
        pos_classes = list(sorted(set(self._matcher_gt.values()) - {0}))
        by: dict[int, list[tuple]] = defaultdict(list)
        for cmp, c in self._matcher_gt.items():
            by[c].append(cmp)
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
        n0_target = max(MIN_PER_CLASS, int(round(self.neg_pos_ratio * total_pos)))

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
        samples.update({0: self._generate_negatives(targets[0], set(self._matcher_gt))})

        d_bal = pl.DataFrame(
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
        self._log.info(
            "balanced: %d, %s",
            len(d_bal),
            ", ".join(
                f"{c}={n}"
                for c, n in d_bal["label"].value_counts(sort=True).iter_rows()
            ),
        )
        self._splits = self._stratified_split(d_bal, [0, *pos_classes])
        return self

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
                label = self._matcher_gt.get(fwd_cmp, self._matcher_gt.get(rev_cmp, 0))
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

    def save(self, output_dir: str, prefix: str = "") -> "SplitGenerator":
        """Write the six CSV files to *output_dir*."""
        if self._splits is None:
            raise RuntimeError(
                "call generate(): attempting to save without generating splits"
            )

        self._log.info("Saving splits to: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for split_name, df in self._splits.items():
            file_name = f"{prefix}_{split_name}.csv" if prefix else f"{split_name}.csv"
            path = os.path.join(output_dir, file_name)
            df.write_csv(path, include_header=True)
            self._log.info("Saved %d rows to file %s", len(df), file_name)
        return self

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
    ) -> list[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]]:
        """Sample *n_needed* cross-cluster negative pairs.

        A candidate pair is accepted iff:
        1. Not the same record.
        2. Different clusters.
        3. Not a known positive.
        4. Not already generated (no duplicates).
        """
        self._log.info("Generating %d negatives", n)

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
        return max(MIN_PER_CLASS, min(target, available))

    def _stratified_split(
        self, df: pl.DataFrame, classes: list[int]
    ) -> dict[str, pl.DataFrame]:
        """Per-class split respecting *split_ratio*, ≥1 per class per split."""
        ratios = list(self.split_ratio.values())
        total_r = sum(ratios)
        parts: dict[str, list] = {k: [] for k in self.split_ratio}

        for c in classes:
            sub = df.filter(pl.col("label") == c)
            n = len(sub)
            assert n >= MIN_PER_CLASS, f"class {c}: {n} < {MIN_PER_CLASS}"

            counts = [max(1, round((n * r_k) / total_r)) for r_k in ratios[1:]]
            n_train = n - sum(counts)

            while n_train < 1 and any(c > 1 for c in counts):
                sorted_counts = sorted(
                    enumerate(counts), key=lambda t: t[1], reverse=True
                )
                best_i, best_c = sorted_counts[0]
                counts[best_i] = best_c - 1
                n_train = n - sum(counts)
            counts.insert(0, n_train)
            assert all(
                c >= 1 for c in counts
            ), f"class {c}: cannot split {n} into {len(counts)} non-empty sets"

            for ix, k in enumerate(parts):
                if ix == 0:
                    parts[k].append(sub[: counts[ix]])
                elif ix < len(parts) - 1:
                    start = sum(counts[:ix])
                    end = start + counts[ix]
                    parts[k].append(sub[start:end])
                else:
                    parts[k].append(sub[sum(counts[:ix]) :])

        return {
            k: pl.concat(v, how="vertical", strict=True).sample(
                fraction=1, seed=self.random_state, shuffle=True, with_replacement=False
            )
            for k, v in parts.items()
        }

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
    ds = CsvDataSource(a.data_source, traits, has_header=True).read()

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
        )
        .load(id_table, cluster_gt, match_gt)
        .generate()
    )
    generator.save(a.output_dir)


if __name__ == "__main__":
    main()
