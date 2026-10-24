from collections import Counter
from dataclasses import dataclass

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._split_generator import SplitGenerator, _class_counts, _pairs_frame

Pair = tuple[RefId, RefId]


@dataclass(frozen=True)
class Bridge:
    pair: Pair  # (target record, bridge record), label 2
    home: int  # cluster containing the bridge record
    proximal: int  # cluster containing the target record


class BadTripletSplitGenerator(SplitGenerator):
    """Each kept bridge's (home, proximal) clusters live in exactly one split.

    A label-2 pair is read as ``(target, bridge)``. A cluster is used by at most
    one kept bridge (as home or proximal), so every kept bridge is an isolated
    two-cluster unit and no graph is needed. Bridges losing a conflict are
    orphaned: their record stays in its home cluster as an ordinary member, or
    is removed when ``drop_orphans`` is set. Kept bridges are chunked by
    ``split_ratio`` quotas; bridges that beat a rival are chunked first, so
    they prefer the train split, and any excess spills into the eval splits.
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
        kept, winners, orphans = self._select_bridges()
        self.orphan_records = {b.pair[1] for b in orphans}
        cluster_split = self._assign_clusters(kept, winners)
        excluded: set[RefId] = set()
        if self.drop_orphans:
            excluded = self.orphan_records - {b.pair[0] for b in kept}
        self._splits = self._build_splits(by, kept, cluster_split, excluded)
        return self

    def _bridges(self) -> list[Bridge]:
        bridges: list[Bridge] = []
        seen: set[RefId] = set()
        for (target, bridge), label in self._matcher_gt.items():
            if label != 2:
                continue
            if bridge in seen:
                raise ValueError(f"bridge record {bridge} has multiple label-2 pairs")
            home = self._cluster_gt.get(bridge)
            proximal = self._cluster_gt.get(target)
            if home is None or proximal is None:
                raise ValueError(
                    f"bridge pair ({target}, {bridge}) lacks cluster labels"
                )
            if home == proximal:
                raise ValueError(f"bridge pair ({target}, {bridge}) joins one cluster")
            seen.add(bridge)
            bridges.append(Bridge((target, bridge), home, proximal))
        return bridges

    def _select_bridges(self) -> tuple[list[Bridge], set[Bridge], list[Bridge]]:
        bridges = self._bridges()
        owner: dict[int, Bridge] = {}  # cluster -> kept bridge using it
        winners: set[Bridge] = set()
        orphans: list[Bridge] = []
        for i in self._rng.permutation(len(bridges)):
            b = bridges[int(i)]
            rival = owner.get(b.home) or owner.get(b.proximal)
            if rival is not None:
                self._log.warning(
                    "dropping bridge %s: clusters (%s, %s) overlap kept bridge %s",
                    b.pair,
                    b.home,
                    b.proximal,
                    rival.pair,
                )
                winners.add(rival)
                orphans.append(b)
                continue
            owner[b.home] = owner[b.proximal] = b
        kept = list(dict.fromkeys(owner.values()))  # dedupe, keep shuffled order
        self._log.info(
            "bridges: %d kept (%d winners), %d orphaned",
            len(kept),
            len(winners),
            len(orphans),
        )
        return kept, winners, orphans

    def _assign_clusters(
        self, kept: list[Bridge], winners: set[Bridge]
    ) -> dict[int, str]:
        names, ratios = list(self.split_ratio), list(self.split_ratio.values())
        units = [b for b in kept if b in winners] + [
            b for b in kept if b not in winners
        ]
        cluster_split: dict[int, str] = {}
        for name, chunk in zip(names, self._chunks(units, ratios, eval_first=False)):
            for b in chunk:
                cluster_split[b.home] = cluster_split[b.proximal] = name
        bridged = {cluster_split[b.home] for b in kept}
        if missing := [n for n in names if n not in bridged]:
            self._log.warning(
                "splits without bridges (label 2 not evaluable): %s", missing
            )

        free = [c for c in self._clusters if c not in cluster_split]
        free = [free[int(i)] for i in self._rng.permutation(len(free))]
        for name, chunk in zip(names, self._chunks(free, ratios)):
            for c in chunk:
                cluster_split[c] = name
        return cluster_split

    def _chunks(
        self, items: list, ratios: list[int], eval_first: bool = True
    ) -> list[list]:
        n, k = len(items), len(ratios)
        if n < k:
            counts = [0] * (k - n) + [1] * n
            if not eval_first:
                counts = [n] + [0] * (k - 1)
        else:
            counts = self._split_counts(n, ratios)
        out, start = [], 0
        for count in counts:
            out.append(items[start : start + count])
            start += count
        return out

    def _cross_cluster_pairs(self, records: list[RefId]) -> int:
        sizes = Counter(self._cluster_gt[r] for r in records)
        n = len(records)
        return (n * n - sum(s * s for s in sizes.values())) // 2

    def _build_splits(
        self,
        by: dict[int, list[Pair]],
        kept: list[Bridge],
        cluster_split: dict[int, str],
        excluded: set[RefId],
    ) -> dict:
        known = set(self._matcher_gt)
        frames = {}
        for name in self.split_ratio:
            records = [
                r
                for r, c in self._cluster_gt.items()
                if cluster_split[c] == name and r not in excluded
            ]
            members = set(records)
            n2 = [b.pair for b in kept if cluster_split[b.home] == name]
            n1 = [p for p in by.get(1, []) if p[0] in members and p[1] in members]
            if n2 and self.match_bridge_ratio > 0:
                cap = round(self.match_bridge_ratio * len(n2))
                n1 = [tuple(p) for p in self._select_diverse(n1, cap)]
            if not n1 and not n2:
                raise RuntimeError(f"split {name} has no positive pairs")
            n0 = min(
                round(self.neg_pos_ratio * (len(n1) + len(n2))),
                self._cross_cluster_pairs(records),
            )
            negatives = self._generate_negatives(n0, known, records) if n0 > 0 else []
            frame = _pairs_frame({0: negatives, 1: n1, 2: n2})
            orphan_count = sum(
                1 for x, y in n1 if x in self.orphan_records or y in self.orphan_records
            )
            self._log.info(
                "split %s: %d records, %s, %d orphan label-1 pairs",
                name,
                len(records),
                _class_counts(frame),
                orphan_count,
            )
            frames[name] = frame
        return frames
