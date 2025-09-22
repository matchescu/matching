"""
Fellegi–Sunter record linkage implementation.

Public API (Arrow-first):
  - FellegiSunter.fit(A: pa.Table, B: pa.Table, truth: pa.Table, truth_label_col: str) -> FSParameters
  - FellegiSunter.estimate_thresholds(A, B, truth, truth_label_col, alpha=0.01, beta=0.01) -> FSThresholds
  - FellegiSunter.score_pairs(A, B, pairs: pa.Table[idA,idB]) -> pa.Table (adds per-field 0/1 and 'score')
  - FellegiSunter.link(A, B, thresholds, block_on: List[str]|None, max_pairs: int|None) -> pa.Table
  - FellegiSunter.parameters_report() -> pa.Table

Notes:
  - Inputs/outputs are Arrow tables. Use `pyarrow.interchange.from_dataframe(obj.__dataframe__())`
    if you want to pass pandas/Polars without converting yourself.
  - Agreement is binary per field: 1 if equal AND both non-null; else 0.
  - Scoring: sum of log-likelihood field weights under independence.
  - Thresholds: empirical quantiles to meet target FMR (alpha) and FNR (beta).

This code favors clarity over ultimate performance. For large-scale linkage, replace
candidate generation with proper blocking/join and vectorize comparisons in batches.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, Sequence, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from matchescu.matching.config import RecordLinkageConfig


@dataclass
class FSComparisonStats:
    match_probability: float
    mismatch_probability: float
    agreement_weight: float
    disagreement_weight: float


@dataclass
class FSParameters:
    comparison_stats: dict[str, FSComparisonStats]
    match_prevalence: float


@dataclass(frozen=True)
class FSThresholds:
    upper: float
    lower: float


# ------------------------------
# Core implementation
# ------------------------------

class FellegiSunter:
    """Canonical Fellegi-Sunter record linkage using exact matches."""
    __L_PREFIX="l_"
    __R_PREFIX="r_"
    __MIN_POSSIBLE_PROB = 1e-12
    __MAX_POSSIBLE_PROB = 1 - 1e-12


    def __init__(self, config: RecordLinkageConfig, mu: float = 0.01, lambda_: float = 0.01):
        self._config = config
        self._cmp_config = self._config.col_comparison_config
        self._params: FSParameters | None = None
        self._thresholds: FSThresholds | None = None
        self._mu: float = mu
        self._lambda: float = lambda_

    def __check_config(self, table_a: pa.Table, table_b: pa.Table, ground_truth: pa.Table | None = None) -> None:
        if self._config.left_id not in table_a.column_names:
            raise KeyError(f"expected ID column '{self._config.left_id}' to be present in the left table")
        if self._config.right_id not in table_b.column_names:
            raise KeyError(f"expected ID column '{self._config.right_id}' to be present in the right table")
        if ground_truth is not None:
            for col in self._config.left_id, self._config.right_id:
                if col not in ground_truth.column_names:
                    raise KeyError(f"expected ID column '{col}' to be present in the ground truth")
            if self._config.ground_truth_label_col not in ground_truth.column_names:
                raise KeyError(f"expected the '{self._config.ground_truth_label_col}' column to be present in the ground truth")
        if self._cmp_config is not None and len(self._cmp_config) > 0:
            for left_col, right_col in self._cmp_config:
                if left_col not in table_a.column_names:
                    raise KeyError(f"expected column '{left_col}' to be present in the left table")
                if right_col not in table_b.column_names:
                    raise KeyError(f"expected column '{right_col}' to be present in the right table")

    def __extract_cmp_config(self, table_a: pa.Table, table_b: pa.Table) -> Sequence[tuple[str, str]]:
        if self._cmp_config is not None and len(self._cmp_config) > 0:
            return self._cmp_config

        cmp_config = []
        for left_col, right_col in zip(table_a.column_names, table_b.column_names):
            if left_col == self._config.left_id or right_col == self._config.right_id:
                continue
            if left_col == right_col:
                cmp_config.append((left_col, right_col))
        if len(cmp_config) == 0:
            raise ValueError("when mappings aren't configured, at least one column must be present in both tables")
        return cmp_config

    @staticmethod
    def __id_to_index_mapping(t: pa.Table, id_col_name: str) -> dict[str, int]:
        return {
            id_: idx
            for idx, id_ in enumerate(t[id_col_name].to_pylist())
        }

    @staticmethod
    def __get_row_dict(t: pa.Table, idx: int) -> dict[str, object]:
        return {name: t[name][idx].as_py() for name in t.column_names}

    @staticmethod
    def __standardize_strings(t: pa.Table) -> pa.Table:
        arrays = []
        schema = []
        for name in t.schema.names:
            col = t[name]
            if pa.types.is_string(col.type):
                cleaned = pc.utf8_trim_whitespace(pc.utf8_lower(col))
                arrays.append(cleaned)
                schema.append((name, cleaned.type))
            else:
                arrays.append(col)
                schema.append((name, col.type))
        return pa.table(arrays, names=[n for n,_ in schema])

    def __is_match(
        self, left_row: dict[str, Any], right_row: dict[str, Any], comparison_idx: int
    ) -> int:
        left_col, right_col = self._cmp_config[comparison_idx]
        left_value, right_value = left_row[left_col], right_row[right_col]

        # use only exact matches to simplify
        return int(left_value == right_value)

    def __cmp_id_col_names(self) -> tuple[str, str]:
        return f"{self.__L_PREFIX}id", f"{self.__R_PREFIX}id"

    @staticmethod
    def __cmp_idx_col_name(idx: int) -> str:
        return f"cmp_{idx}"

    def __cmp_config_member_cols(self) -> Iterable[str]:
        for left_col, right_col in self._cmp_config:
            yield f"{self.__L_PREFIX}{left_col}"
            yield f"{self.__R_PREFIX}{right_col}"

    def __get_comparison_table(self, id_pairs: Iterable[tuple[Any, Any]], table_a: pa.Table, table_b: pa.Table) -> pa.Table:
        lid_col, rid_col = self.__cmp_id_col_names()
        ltable_idx_map = self.__id_to_index_mapping(table_a, self._config.left_id)
        rtable_idx_map = self.__id_to_index_mapping(table_b, self._config.right_id)
        comparison_rows = [
            (
                self.__get_row_dict(table_a, ltable_idx_map[left_id]),
                self.__get_row_dict(table_b, rtable_idx_map[right_id])
            )
            for left_id, right_id in id_pairs
        ]

        # Build comparison rows in Python (simple & clear).
        rows = []
        for left_row, right_row in comparison_rows:
            comparison_result = {
                lid_col: left_row[self._config.left_id],
                rid_col: right_row[self._config.right_id],
            }
            for left_col, right_col in self._cmp_config:
                comparison_result.update({
                    f"{self.__L_PREFIX}{left_col}": left_row[left_col],
                    f"{self.__R_PREFIX}{right_col}": right_row[right_col]
                })
            comparison_result.update({
                self.__cmp_idx_col_name(cmp_idx): self.__is_match(left_row, right_row, cmp_idx)
                for cmp_idx in range(len(self._cmp_config))
            })
            rows.append(comparison_result)

        cmp_cols = list(map(self.__cmp_idx_col_name, range(len(self._cmp_config))))
        cmp_member_cols = list(self.__cmp_config_member_cols())
        names = [lid_col, rid_col, *cmp_member_cols, *cmp_cols]

        if not rows:
            return pa.table({n: pa.array([], pa.null()) for n in names})

        # Arrow-ify
        cols = {k: [r[k] for r in rows] for k in names}
        # Field cols are ints; id cols inherit inferred types
        for cmp_col_name in cmp_cols:
            cols[cmp_col_name] = pa.array(cols[cmp_col_name], type=pa.int8())
        return pa.table(cols)

    # ---------- Fit (m/u and weights) ----------

    def fit(self, table_a: pa.Table, table_b: pa.Table, ground_truth: pa.Table, *, smooth: float = 1e-6) -> "FellegiSunter":
        """
        Estimate m_i = P(agree on field i | M), u_i = P(agree on field i | U)
        and compute per-field log-weights.
        """
        self.__check_config(table_a, table_b, ground_truth)

        table_a = self.__standardize_strings(table_a)
        table_b = self.__standardize_strings(table_b)
        id_pair_list = list(zip(ground_truth[self._config.left_id].to_pylist(), ground_truth[self._config.right_id].to_pylist()))
        cmp_table = self.__get_comparison_table(id_pair_list, table_a, table_b)
        cmp_result_cols = list(map(self.__cmp_idx_col_name, range(len(self._cmp_config))))

        labels = np.asarray(ground_truth[self._config.ground_truth_label_col].to_pylist())
        real_matches = labels == 1
        real_mismatches = labels == 0

        matches = np.asarray(real_matches, dtype=int)
        real_match_count_adj = matches.sum() + 2 * smooth
        real_mismatch_count_adj = np.asarray(real_mismatches, dtype=int).sum() + 2 * smooth

        cmp_stats: dict[str, FSComparisonStats] = {}
        for col_name in cmp_result_cols:
            cmp_results = np.asarray(cmp_table[col_name].to_numpy())  # 0/1 ints
            col_matches_in_real_matches = cmp_results[real_matches].sum() + smooth
            col_matches_in_real_mismatches = cmp_results[real_mismatches].sum() + smooth
            m_prob = col_matches_in_real_matches / real_match_count_adj  # agree
            u_prob = col_matches_in_real_mismatches / real_mismatch_count_adj  # P(scenario| real

            # make sure m_prob and u_prob are different from 0
            m_prob = float(np.clip(m_prob, self.__MIN_POSSIBLE_PROB, self.__MAX_POSSIBLE_PROB))
            u_prob = float(np.clip(u_prob, self.__MIN_POSSIBLE_PROB, self.__MAX_POSSIBLE_PROB))

            agreement_factor = m_prob / u_prob  # how much more likely something is to match than not match
            disagreement_factor = (1-m_prob) / (1-u_prob)  # how much more likely is to not match than to match

            cmp_stats[col_name] = FSComparisonStats(
                m_prob,
                u_prob,
                np.log2(agreement_factor), # match agreement weight
                np.log2(disagreement_factor) # match disagreement weight
            )

        self._params = FSParameters(cmp_stats, float(matches.mean()))
        scored_cmp_table = self.__compute_scores(cmp_table)
        self._thresholds = self.__compute_thresholds(scored_cmp_table, labels)
        return self

    def __compute_row_score(self, agree_vec: Dict[str, int]) -> float:
        s = 0.0
        for col_name in agree_vec:
            col_name_stats = self._params.comparison_stats[col_name]
            s += col_name_stats.agreement_weight if agree_vec[col_name] == 1 else col_name_stats.disagreement_weight
        return float(s)

    def __compute_scores(self, cmp_table: pa.Table) -> pa.Table:
        assert self._params is not None, "training params not initialised."
        scores = [
            self.__compute_row_score(
                {
                    col: cmp_table[col][i].as_py()
                    for col in map(self.__cmp_idx_col_name, range(len(self._cmp_config)))
                }
            )
            for i in range(len(cmp_table))
        ]
        out = cmp_table.append_column("score", pa.array(scores, type=pa.float64()))
        return out

    def __compute_thresholds(self, scored_table: pa.Table, labels: np.ndarray) -> FSThresholds:
        scores = np.asarray(scored_table["score"].to_numpy())
        match_scores = scores[labels == 1]
        mismatch_scores = scores[labels == 0]

        upper = float(np.quantile(mismatch_scores, 1 - self._mu))
        lower = float(np.quantile(match_scores, self._lambda))
        if lower > upper:
            mid = float((lower + upper) / 2.0)
            lower = upper = mid
        return FSThresholds(upper=upper, lower=lower)

    def __decide(self, s: float, th: FSThresholds) -> str:
        if s >= th.upper:
            return "link"
        elif s <= th.lower:
            return "non-link"
        else:
            return "clerical"

    def predict(self, id_pairs: Iterable[tuple[Any, Any]], table_a: pa.Table, table_b: pa.Table) -> pa.Table:
        """
        Link records using learned parameters and thresholds.

        Candidate generation:
          - if block_on is provided: exact-equality blocking on those columns (simple Python join)
          - else: Cartesian product (O(n*m), for small cases)

        Returns Arrow table with per-field agreements, score, and decision.
        """
        assert self._params is not None and self._thresholds is not None, "model not fit. run fit() first."
        table_a = self.__standardize_strings(table_a)
        table_b = self.__standardize_strings(table_b)
        cmp_table = self.__get_comparison_table(id_pairs, table_a, table_b)
        scored = self.__compute_scores(cmp_table)
        decisions = [self.__decide(s, self._thresholds) for s in scored["score"].to_pylist()]
        out = scored.append_column("decision", pa.array(decisions, type=pa.string()))
        # sort by score desc
        order = np.argsort(-np.asarray(out["score"].to_numpy()))
        # take rows in desired order
        return out.take(pa.array(order, type=pa.int64()))
