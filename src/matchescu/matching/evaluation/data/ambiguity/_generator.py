import re

from difflib import SequenceMatcher
from functools import reduce, partial
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable, Set

import networkx as nx
import polars as pl

from matchescu.data import Record
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import Traits, RecordExtraction, single_record
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import (
    EntityReferenceIdentifier as RefId, EntityReference,
)

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None


DIR = Path("./data/affiliationstrings/")
DEFAULT_BRIDGE_TERMS: list[str] = [
    "Research", "Center", "Lab", "Institute", "College",
    "Department", "Division", "Group", "School", "Foundation",
]


type ComparisonData = tuple[RefId, RefId]


class AmbiguityGenerator:
    _DEFAULT_DEGRADATION_PATTERNS = [
        r',\s*USA', r',\s*US', r',\s*United States',
        r',\s*CA', r',\s*California',
        r',\s*NY', r',\s*New York',
        r',\s*Germany', r',\s*France', r',\s*Japan',
        r'\s+Inc\.?', r'\s+Corp\.?', r'\s+Ltd\.?',
        r'\s+University', r'\s+Univ\.?',
        r'\s+Research\s+Center', r'\s+Lab(?:oratories)?'
    ]

    def __init__(
        self,
        data_sources: list[CsvDataSource],
        mapping_gt: dict[ComparisonData, int],
        ambiguity_target_properties: list[str] = None,
        string_degradation_patterns: list[str] = None,
        id_col: str|int = 0,
    ) -> None:
        self._mapping_gt = mapping_gt
        self._id_col = id_col
        self._id_table = reduce(self.__ingest_all, data_sources, InMemoryIdTable())
        self._cluster_id_map, self._cluster_count = self.__get_clusters()
        self._degradation_patterns = string_degradation_patterns or self._DEFAULT_DEGRADATION_PATTERNS
        self._target_properties = set(ambiguity_target_properties or [])

    def __new_ref_id(self, records: Iterable[Record], source: str):
        dominant_record = next(iter(records))
        return RefId(label=dominant_record[self._id_col], source=source)

    def __ingest_all(self, id_table: InMemoryIdTable, data_source: CsvDataSource) -> InMemoryIdTable:
        id_factory = partial(self.__new_ref_id, source=data_source.name)
        extract_entity_references = RecordExtraction(data_source, id_factory, single_record)
        for ref in extract_entity_references():
            id_table.put(ref)
        return id_table

    def __get_clusters(self):
        ref_ids = list(self._mapping_gt)
        g = nx.DiGraph(ref_ids)
        cluster_count = 0
        cluster_ref_map = {}

        for cluster_count, cluster in enumerate(nx.strongly_connected_components(g), 1):
            if cluster_count not in cluster_ref_map:
                cluster_ref_map[cluster_count] = set()
            for ref_id in cluster:
                cluster_ref_map[cluster_count].add(ref_id)

        return cluster_ref_map, cluster_count

    def degrade_str(self, input_str):
        result = input_str
        for pattern in self._degradation_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Clean up extra spaces and commas
        result = re.sub(r'\s*,\s*,', ', ', result)
        result = re.sub(r'^\s*,\s*|\s*,\s*$', '', result)
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _select_cluster_representative(self, cluster, selector) -> str:
        prev_degraded = ""
        for ref in cluster:
            initial_value = selector(ref)
            degraded = self.degrade_str(initial_value)
            if len(degraded) <= 5:
                continue
            if degraded.lower() == initial_value.lower():
                continue
            if len(degraded) < len(prev_degraded) or len(prev_degraded) == 0:
                prev_degraded = degraded
        return prev_degraded

    @classmethod
    def _edit_distance(cls, s1: str, s2: str) -> int:
        if Levenshtein is not None:
            return Levenshtein.distance(s1, s2)
        if len(s1) < len(s2):
            return cls._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, 1 otherwise
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @classmethod
    def _edit_similarity(cls, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0

        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0 if s1 == s2 else 0.0

        distance = cls._edit_distance(s1, s2)

        return 1.0 - (distance / max_len)

    def _ambiguity_score(self, x: EntityReference, y: EntityReference) -> float:
        """Compute how ambiguous the result of combining ``x`` and ``y`` would be.

        The algorithm for computing ambiguity uses all of the property names provided
        by the user to compare x and y. For each property value it applies the
        following rules:

        - identical property values get +2.0
        - 2.0 * inv(edit distance) if property values are not identical
        - property is not a string gets -0.5
        - type mismatch gets -1.0
        - property does not exist in ``x`` or ``y`` other gets -2.0

        :param x: first entity reference
        :param y: second entity reference
        :params property_names: variable array of property names to consider when
        computing ambiguity; these are the properties that would likely yield the
        ambiguous values.

        :return: an ambiguity score: the higher, the more chances of being able to
        generate an ambiguous entity reference.
        """
        score = 0.0

        x_dict = x.as_dict()
        y_dict = y.as_dict()
        all_keys = set(x_dict.keys()) | set(y_dict.keys())

        for key in all_keys:
            if (key in x_dict) ^ (key in y_dict):
                # the ambiguous reference will need to remove differentiators
                score -= 0.5
                continue

            val_a = x_dict[key]
            val_b = y_dict[key]

            if val_a == val_b:
                # all identical values are wonderful for creating ambiguity
                score += 2.0
            elif isinstance(val_a, str) and isinstance(val_b, str):
                val_a = val_a.lower().strip()
                val_b = val_b.lower().strip()
                if key in self._target_properties:
                    # obtaining an ambiguous entity reference is still desirable
                    # when the key is part of the targeted properties
                    increment = 0
                    if val_a in val_b or val_b in val_a:
                        increment = 2.0
                    else:
                        increment = 2.0 * self._edit_similarity(val_a, val_b)
                    score += increment
                else:
                    # non-target different strings become 'None' (information loss)
                    score -= 0.5
            else:
                # Type mismatch or non-string difference results in 'None', but the
                # matching model would probably have an easier time discriminating
                # if there's a type mismatch between the bridged entity references
                score -= 1.0

        return score

    def _get_cluster_representatives(self):
        cluster_reps = {}
        for cluster_id, ref_ids in self._cluster_id_map.items():
            candidates = list(ref_ids)
            refs = list(self._id_table.get_all(candidates))
            mean_ambiguities = []
            for i, ref in enumerate(refs):
                others = refs[:i] + refs[i+1:]
                ambiguity_scores = [self._ambiguity_score(ref, x) for x in others]
                mean_ambiguities.append(sum(ambiguity_scores) / len(ambiguity_scores))
            max_ambiguity = 0
            rep_idx = 0
            for i, ambiguity in enumerate(mean_ambiguities):
                if max_ambiguity < ambiguity:
                    max_ambiguity = ambiguity
                    rep_idx = i
            cluster_reps[cluster_id] = candidates[rep_idx]
        return cluster_reps

    def _find_closest_rep(self, cluster_reps):
        cluster_refs = {
            cluster_id: self._id_table.get(ref_id)
            for cluster_id, ref_id in cluster_reps.items()
        }
        result = {}
        for cluster_id, ref in cluster_refs.items():
            if cluster_id in result:
                continue
            other_refs = {
                other_id: other_ref
                for other_id, other_ref in cluster_refs.items()
                if other_id != cluster_id and other_id not in result
            }
            if len(other_refs) < 1:
                break

            best_ambiguity = None
            best_id = None
            for other_id, other_ref in other_refs.items():
                score = self._ambiguity_score(ref, other_ref)
                if best_ambiguity is None or score > best_ambiguity:
                    best_ambiguity = score
                    best_id = other_id
            result[cluster_id] = (best_id, best_ambiguity, other_refs[best_id])
        return result

    @classmethod
    def _ngram_sim(cls, a: str, b: str, n: int = 3) -> float:
        """Character n-gram Jaccard similarity."""
        a, b = a.lower(), b.lower()
        if len(a) < n or len(b) < n:
            sa, sb = set(a.replace(" ", "")), set(b.replace(" ", ""))
            return len(sa & sb) / max(len(sa | sb), 1)
        na = {a[i : i + n] for i in range(len(a) - n + 1)}
        nb = {b[i : i + n] for i in range(len(b) - n + 1)}
        return len(na & nb) / max(len(na | nb), 1)

    @classmethod
    def _tok_sim(cls, a: str, b: str) -> float:
        """Word-level Jaccard similarity."""
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(len(wa | wb), 1)

    @classmethod
    def _weighted_similarity(cls, a: str, b: str) -> float:
        """Weighted composite: 40 % char-ngram, 30 % token, 30 % sequence."""
        return (
            0.4 * cls._ngram_sim(a, b)
            + 0.3 * cls._tok_sim(a, b)
            + 0.3 * SequenceMatcher(None, a.lower(), b.lower()).ratio()
        )

    @classmethod
    def _score(cls, cand: str, a: str, b: str) -> tuple[float, float, float]:
        """(harmonic_mean, sim_a, sim_b) — 0 if cand equals either input."""
        if cand.lower() in (a.lower(), b.lower()):
            return 0.0, 0.0, 0.0
        sa, sb = cls._weighted_similarity(cand, a), cls._weighted_similarity(cand, b)
        h = 2 * sa * sb / (sa + sb) if sa + sb else 0.0
        return h, sa, sb

    @classmethod
    def _cpfx(cls, a: str, b: str) -> str:
        """Longest common case-insensitive prefix, preserving case from *a*."""
        i = 0
        for x, y in zip(a.lower(), b.lower()):
            if x != y:
                break
            i += 1
        return a[:i]

    @classmethod
    def _blend(cls, a: str, b: str) -> str:
        """Alternate characters: even indices from *a*, odd from *b*."""
        return "".join(
            (a[i] if i % 2 == 0 else b[i])
            if i < len(a) and i < len(b)
            else (a[i] if i < len(a) else b[i])
            for i in range(max(len(a), len(b)))
        )

    @classmethod
    def _candidates(cls, a: str, b: str, bridge: list[str]) -> set[str]:
        wa, wb = a.split(), b.split()
        la, lb = {w.lower() for w in wa}, {w.lower() for w in wb}
        cs = la & lb

        common = [w for w in wa if w.lower() in cs]
        ua = [w for w in wa if w.lower() not in cs]
        ub = [w for w in wb if w.lower() not in cs]
        base = " ".join(common)

        pool: set[str] = set()

        def _add(s: str) -> None:
            s = " ".join(s.split())
            if s and s != a and s != b:
                pool.add(s)

        # 1. Base + every prefix-truncation of each unique word
        if base:
            _add(base)
        for w in ua + ub:
            for k in range(1, len(w)):
                _add(f"{base} {w[:k]}")

        # 2. Pair unique words -> common-prefix extensions / blend / mid-cut
        for wa_ in ua:
            for wb_ in ub:
                cp = cls._cpfx(wa_, wb_)
                if cp:
                    _add(f"{base} {cp}")
                    for e in range(1, max(len(wa_), len(wb_)) - len(cp) + 1):
                        for w in (wa_, wb_):
                            if len(cp) + e <= len(w):
                                _add(f"{base} {w[: len(cp) + e]}")
                _add(f"{base} {cls._blend(wa_, wb_)}")
                mid = (len(wa_) + len(wb_)) // 2
                if mid:
                    _add(f"{base} {wa_[:mid]}")
                    _add(f"{base} {wb_[:mid]}")

        # 3. Cross-structural mixing (comma-separated hybrids)
        if ua and ub:
            _add(f"{', '.join(ua)}, {b}")
            _add(f"{', '.join(ub)}, {a}")
            for x in ua[:2]:
                for y in ub[:2]:
                    _add(f"{x}, {' '.join(common + [y])}")
                    _add(f"{y}, {' '.join(common + [x])}")

        # 4. Word substitution inside each input's frame
        for i, w in enumerate(wa):
            for ww in wb:
                if w.lower() != ww.lower():
                    _add(" ".join(wa[:i] + [ww] + wa[i + 1:]))
        for i, w in enumerate(wb):
            for ww in wa:
                if w.lower() != ww.lower():
                    _add(" ".join(wb[:i] + [ww] + wb[i + 1:]))

        # 5. Subsets & (small) permutations of the word union
        union = list(dict.fromkeys(wa + wb))
        cap = min(len(union) + 1, 5)
        for r in range(1, cap):
            for c in combinations(union, r):
                _add(" ".join(c))
            if len(union) <= 6:
                for p in permutations(union, r):
                    _add(" ".join(p))

        # 6. Domain bridge terms
        for t in bridge:
            if base:
                _add(f"{base} {t}")
            for w in (ua + ub)[:2]:
                _add(f"{w} {t}")

        # 7. Containment: gradual extension of the shorter string
        lo_a, lo_b = a.lower(), b.lower()
        if lo_a.startswith(lo_b) or lo_b.startswith(lo_a):
            longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
            extra = longer[len(shorter):].strip()
            for k in range(1, len(extra) + 1):
                _add(f"{shorter} {extra[:k]}")

        return pool

    @classmethod
    def _generate_ambiguous_text(
        cls, val_a: str, val_b: str, n: int = 5, bridge_terms: list[str] | None = None,
    ) -> list[tuple[str, float, float, float]]:
        """Return the *n* most ambiguous strings between *val_a* and *val_b*.

        Returns
        -------
        list of (candidate, ambiguity_score, sim_to_a, sim_to_b)
            Sorted by *ambiguity_score* (harmonic mean of both similarities),
            highest first.
        """
        bt = bridge_terms if bridge_terms is not None else DEFAULT_BRIDGE_TERMS
        pool = cls._candidates(val_a, val_b, bt)
        ranked = sorted(
            ((c, *cls._score(c, val_a, val_b)) for c in pool),
            key=lambda r: r[1],
            reverse=True,
        )
        return ranked[:n]

    def _generate_ambiguous_references(self, cluster_reps, closest_correspondents):
        max_id = max(map(lambda r: r.id.label, self._id_table))
        start_refs = {k: self._id_table.get(v) for k, v in cluster_reps.items()}

        ambiguous_refs = {}
        for cluster_id, start_ref in start_refs.items():
            if cluster_id not in closest_correspondents:
                continue
            target_cluster_id, score, target = closest_correspondents[cluster_id]
            source_dict = start_ref.as_dict()
            target_dict = target.as_dict()
            all_keys = set(source_dict.keys()) | set(target_dict.keys())
            ref_properties = {
                "source": start_ref,
                "target": target,
                "target_cluster_id": target_cluster_id
            }
            for k in all_keys:
                # attribute in one but not the other -> continue
                if (k in source_dict) ^ (k in target_dict):
                    continue
                val_a = start_ref[k]
                val_b = target[k]
                if val_a == val_b:
                    ref_properties[k] = val_a
                elif isinstance(val_a, str) and isinstance(val_b, str):
                    if k not in self._target_properties:
                        ref_properties[k] = ""
                    else:
                        ref_properties[k] = self._generate_ambiguous_text(val_a, val_b, 1)[0][0]
                else:
                    ref_properties[k] = None
            new_ref = EntityReference(RefId(max_id + 1, "generated"), ref_properties)
            self._id_table.put(new_ref)
            ambiguous_refs[cluster_id] = new_ref.id
            max_id += 1
        return ambiguous_refs

    def __call__(self) -> tuple[
        InMemoryIdTable,
        dict[int, Set[RefId]],
        dict[tuple[RefId, RefId], int],
        dict[tuple[RefId, RefId], int],
    ]:
        cluster_reps = self._get_cluster_representatives()
        closest_reps = self._find_closest_rep(cluster_reps)
        ambiguous_refs = self._generate_ambiguous_references(cluster_reps, closest_reps)

        cluster_gt: dict[int, set[RefId]] = self._cluster_id_map.copy()
        for cluster_id, ref in ambiguous_refs.items():
            cluster_gt[cluster_id].add(ref)

        directed_gt: dict[ComparisonData, int] = {}
        undirected_gt: dict[ComparisonData, int] = {}
        for cluster_id, ref_ids in cluster_gt.items():
            for cmp_data in combinations(ref_ids, 2):
                undirected_gt[cmp_data] = 1
                directed_gt[cmp_data] = 1
            for ref in self._id_table.get_all(ref_ids):
                if not hasattr(ref, "target_cluster_id"):
                    continue
                fwd_comparison = (ref.id, ref.target.id)
                rev_comparison = (ref.target.id, ref.id)
                undirected_gt[fwd_comparison] = 1
                directed_gt[fwd_comparison] = 3
                directed_gt[rev_comparison] = 2

        return self._id_table, cluster_gt, directed_gt, undirected_gt



def load_data(datadir, rec_fname, traits, mapping_fname):
    ds = CsvDataSource(datadir / rec_fname, traits, has_header=True)
    ds.read()
    df = pl.read_csv(
        datadir / mapping_fname,
        has_header=False,
        schema={"left_id": pl.Int32, "right_id": pl.Int32}
    )
    mappings = {
        (
            RefId(label=row["left_id"], source=ds.name),
            RefId(label=row["right_id"], source=ds.name)
        ): 1
        for row in df.iter_rows(named=True)
    }
    return ds, mappings


def main():
    traits = list(Traits().string(["affil1"]))
    ds, matching_gt = load_data(DIR, "affiliationstrings_ids.csv", traits, "affiliationstrings_mapping.csv")
    introduce_ambiguity = AmbiguityGenerator([ds], matching_gt, ["affil1"], None, "id1")
    id_table, cluster_gt, directed_gt, undirected_gt = introduce_ambiguity()

    data_df = pl.DataFrame([
        {
            "id1": ref.id.label,
            "source": ref.id.source,
            "affil1": ref.affil1,
        }
        for ref in id_table
    ])
    data_df.write_csv(DIR / f"{ds.name}_with_ambiguity.csv", include_header=True)

    clusters_df = pl.DataFrame([
        {
            "id": ref_id.label,
            "source": ref_id.source,
            "cluster_id": cluster_id
        }
        for cluster_id, ref_ids in cluster_gt.items()
        for ref_id in ref_ids
    ])
    directed_df = pl.DataFrame([
        {
            "left_id": l.label,
            "left_source": l.source,
            "right_id": r.label,
            "right_source": r.source,
            "label": c
        }
        for (l, r), c in directed_gt.items()
    ])
    undirected_df = pl.DataFrame([
        {
            "left_id": l.label,
            "left_source": l.source,
            "right_id": r.label,
            "right_source": r.source,
            "label": c
        }
        for (l, r), c in undirected_gt.items()
    ])

    ground_truths = {
        "affiliationstrings_clusters": clusters_df,
        "affiliationstrings_directed": directed_df,
        "affiliationstrings_undirected": undirected_df,
    }

    for name, df in ground_truths.items():
        df.write_csv(DIR / f"{name}_amb_gt.csv", include_header=True)


if __name__ == "__main__":
    main()