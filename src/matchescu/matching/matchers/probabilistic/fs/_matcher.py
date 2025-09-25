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

import pickle
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Sequence, Any, Optional, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from matchescu.matching.config import RecordLinkageConfig
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId, EntityReference


@dataclass(frozen=True)
class FSComparisonStats:
    level_values: np.ndarray  # shape (K,), e.g. np.array([0,1]) or np.array([-1,0,1])
    m_probs: np.ndarray  # shape (K,)
    u_probs: np.ndarray  # shape (K,)
    weights: np.ndarray  # shape (K,)
    level_to_index: dict[Any, int]  # maps raw outcome -> index into arrays


@dataclass(frozen=True)
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


def eq_casefold(a: str, b: str) -> Literal[0, 1]:
    if a is None or b is None:
        return 0
    a = str(a).casefold().strip()
    b = str(b).casefold().strip()
    return 1 if a == b else 0


class FellegiSunter:
    """Canonical Fellegi-Sunter record linkage using exact matches."""

    __L_PREFIX = "l_"
    __R_PREFIX = "r_"
    __NEAR_ZERO = 1e-12
    __NEAR_ONE = 1 - 1e-12

    def __init__(
        self,
        config: RecordLinkageConfig,
        mu: float = 0.01,
        lambda_: float = 0.01,
        parameters: Optional[FSParameters] = None,
        thresholds: Optional[FSThresholds] = None,
    ):
        self._config = config
        self._cmp_config = self._config.col_comparison_config
        self._params = parameters
        self._thresholds = thresholds
        self._mu: float = mu
        self._lambda: float = lambda_

    @property
    def config(self) -> RecordLinkageConfig:
        return self._config

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def lambda_(self) -> float:
        return self._lambda

    @property
    def thresholds(self) -> FSThresholds:
        return self._thresholds

    @property
    def parameters(self) -> FSParameters:
        return self._params

    def fit(
        self,
        comparison_space: BinaryComparisonSpace,
        id_table: IdTable,
        ground_truth: set[tuple[RefId, RefId]],
        smooth: float = 1e-6,
    ) -> "FellegiSunter":
        """Compute the parameters and thresholds defined in the F-S model.

        :param comparison_space: a list of pairs of IDs to compare
        :param id_table: a lookup table which maps IDs to entity references
        :param ground_truth: a set of ID pairs that constitute observed good
            matches
        :param smooth: smoothing factor (helps with division by zero)

        :return: the current instance of the matcher with 'trained' parameters
        and thresholds. Enables the user to write fluent code such as
        ``FellegiSunter(config).fit(train_a, train_b, truth).predict(comparisons, test_a, test_b)``.
        """

        assert len(self._cmp_config) > 0
        id_pairs = list(comparison_space)
        cmp_table = self.__get_comparison_table(comparison_space, id_table)
        labels = np.asarray([int(pair in ground_truth) for pair in id_pairs])
        M = labels == 1
        U = labels == 0

        cmp_stats: dict[str, FSComparisonStats] = {}
        for i, config in enumerate(self._cmp_config):
            col_name = self.__cmp_idx_col_name(i)
            cmp_results = np.asarray(cmp_table[col_name].to_numpy())

            K = len(config.agreement_levels)
            # count the occurrence of each level within real links and real non-links
            counts_m = np.array([np.sum(cmp_results[M] == v) for v in config.agreement_levels])
            counts_u = np.array([np.sum(cmp_results[U] == v) for v in config.agreement_levels])
            # compute likelihoods of each level within real links and real non-links
            clip = partial(np.clip, a_min=self.__NEAR_ZERO, a_max=self.__NEAR_ONE)
            m_probs = clip((counts_m + smooth) / (counts_m.sum() + K * smooth))
            u_probs = clip((counts_u + smooth) / (counts_u.sum() + K * smooth))
            # how much more likely is it for each level to appear in M vs. U
            bayesian_factors = m_probs / u_probs
            # weights are easier to work with (sums, no overflows)
            weights = np.log2(bayesian_factors)
            level_to_index = {v: i for i, v in enumerate(config.agreement_levels)}
            cmp_stats[col_name] = FSComparisonStats(
                np.array(config.agreement_levels), m_probs, u_probs, weights, level_to_index
            )

        self._params = FSParameters(
            cmp_stats, float((labels == 1).astype(dtype=np.int8).mean())
        )
        scored_cmp_table = self.__compute_scores(cmp_table)
        self._thresholds = self.__compute_thresholds(scored_cmp_table, labels)
        return self

    def predict(
        self, id_pairs: BinaryComparisonSpace, id_table: IdTable
    ) -> set[tuple[Any, Any]]:
        """Link records and return a set of linked (left_id_label, right_id_label) using learned parameters.

        The records being linked are identified using the supplied ``id_pairs``.
        The pairs of IDs are typically obtained through candidate generation
        techniques (blocking, filtering, etc.). If an ID does not exist in
        either table, a ``KeyError`` is raised and the process fails completely.

        :param id_pairs: an iterable sequence of pairs of record identifiers.
        The first pair member identifies a record in ``table_a`` whereas the
        second identifies a record in ``table_b``.
        :param id_table: mapping from ID to entity reference

        :return: a set of linked (left_id_label, right_id_label) pairs.
        """
        assert (
            self._params is not None and self._thresholds is not None
        ), "model not fit. run fit() first."
        cmp_table = self.__get_comparison_table(id_pairs, id_table)
        scored = self.__compute_scores(cmp_table)
        decisions = [
            self.__decide(s, self._thresholds) for s in scored["score"].to_pylist()
        ]

        result = set()
        for i, decision in enumerate(decisions):
            if decision != "link":
                continue
            result.add((cmp_table["l_id"][i].as_py(), cmp_table["r_id"][i].as_py()))
        return result

    def save(self, filename: str) -> None:
        assert (
            self._params is not None and self._thresholds is not None
        ), "model not trained. run fit() before saving."

        with open(filename, "wb") as f:
            saved_data = (
                self._params,
                self._thresholds,
                self._mu,
                self._lambda,
                self._config,
            )
            pickle.dump(saved_data, f)

    def __check_config(
        self, table_a: pa.Table, table_b: pa.Table, ground_truth: pa.Table | None = None
    ) -> None:
        if self._config.left_id not in table_a.column_names:
            raise KeyError(
                f"expected ID column '{self._config.left_id}' to be present in the left table"
            )
        if self._config.right_id not in table_b.column_names:
            raise KeyError(
                f"expected ID column '{self._config.right_id}' to be present in the right table"
            )
        if self._cmp_config is not None and len(self._cmp_config) > 0:
            for left_col, right_col in self._cmp_config:
                if left_col not in table_a.column_names:
                    raise KeyError(
                        f"expected column '{left_col}' to be present in the left table"
                    )
                if right_col not in table_b.column_names:
                    raise KeyError(
                        f"expected column '{right_col}' to be present in the right table"
                    )

    def __init_cmp_config(
        self, table_a: pa.Table, table_b: pa.Table
    ) -> Sequence[tuple[str, str]]:
        if self._cmp_config is not None and len(self._cmp_config) > 0:
            return self._cmp_config
        left_cols = [c for c in table_a.column_names if c != self._config.left_id]
        right_set = set(c for c in table_b.column_names if c != self._config.right_id)
        common_cols = [c for c in left_cols if c in right_set]
        if len(common_cols) == 0:
            raise ValueError(
                "when mappings aren't configured, at least one column must be present in both tables"
            )
        return [(c, c) for c in common_cols]

    @staticmethod
    def __id_to_index_mapping(t: pa.Table, id_col_name: str) -> dict[Any, int]:
        return {id_: idx for idx, id_ in enumerate(t[id_col_name].to_pylist())}

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
        return pa.table(arrays, names=[n for n, _ in schema])

    def __compute_similarity(
        self, left_row: EntityReference, right_row: EntityReference, comparison_idx: int
    ) -> float:
        attr_cmp = self._cmp_config[comparison_idx]
        left_value, right_value = left_row[attr_cmp.left], right_row[attr_cmp.right]

        return attr_cmp.sim(left_value, right_value)

    def __cmp_id_col_names(self) -> tuple[str, str]:
        return f"{self.__L_PREFIX}id", f"{self.__R_PREFIX}id"

    @staticmethod
    def __cmp_idx_col_name(idx: int) -> str:
        return f"cmp_{idx}"

    def __cmp_config_fields(
        self, table_a: pa.Table, table_b: pa.Table
    ) -> Iterable[pa.Field]:
        for attr_cmp in self._cmp_config:
            yield table_a.schema.field(attr_cmp.left).with_name(
                f"{self.__L_PREFIX}{attr_cmp.left}"
            )
            yield table_b.schema.field(attr_cmp.right).with_name(
                f"{self.__R_PREFIX}{attr_cmp.right}"
            )

    def __get_comparison_table(
        self, id_pairs: Iterable[tuple[Any, Any]], id_table: IdTable
    ) -> pa.Table:
        lid_col, rid_col = self.__cmp_id_col_names()
        comparison_rows = list(map(id_table.get_all, id_pairs))

        # Build comparison rows in Python (simple & clear).
        rows = []
        for left_row, right_row in comparison_rows:
            comparison_result = {
                lid_col: left_row.id.label,
                rid_col: right_row.id.label,
            }
            for attr_cmp in self._cmp_config:
                comparison_result.update(
                    {
                        f"{self.__L_PREFIX}{attr_cmp.left}": left_row[attr_cmp.left],
                        f"{self.__R_PREFIX}{attr_cmp.right}": right_row[attr_cmp.right],
                    }
                )
            comparison_result.update(
                {
                    self.__cmp_idx_col_name(cmp_idx): self.__compute_similarity(
                        left_row, right_row, cmp_idx
                    )
                    for cmp_idx in range(len(self._cmp_config))
                }
            )
            rows.append(comparison_result)

        all_keys = list({k: None for r in rows for k in r}.keys())
        columnar_mapping = {k: [row.get(k) for row in rows] for k in all_keys}

        return pa.table(columnar_mapping)

    def __compute_row_score(self, level_indices_in_row: Dict[str, int]) -> float:
        s = 0.0
        for col_name, raw_level in level_indices_in_row.items():
            stats = self._params.comparison_stats[col_name]
            try:
                idx = stats.level_to_index[raw_level]
            except KeyError:
                raise KeyError(
                    f"Unseen level {raw_level!r} for comparison {col_name!r}. "
                    f"Known levels: {stats.level_values.tolist()}"
                ) from None
            s += float(stats.weights[idx])
        return float(s)

    def __compute_scores(self, cmp_table: pa.Table) -> pa.Table:
        assert self._params is not None, "training params not initialised."
        cmp_result_cols = list(
            map(self.__cmp_idx_col_name, range(len(self._cmp_config)))
        )
        scores = [
            self.__compute_row_score(
                {col: cmp_table[col][i].as_py() for col in cmp_result_cols}
            )
            for i in range(len(cmp_table))
        ]
        out = cmp_table.append_column("score", pa.array(scores, type=pa.float64()))
        return out

    def __compute_thresholds(
        self, scored_table: pa.Table, labels: np.ndarray
    ) -> FSThresholds:
        scores = np.asarray(scored_table["score"].to_numpy())
        msk_m = labels == 1
        msk_u = ~msk_m
        match_scores = scores[msk_m]
        mismatch_scores = scores[msk_u]

        def _q(a: np.ndarray, q: float) -> float:
            if a.size == 0:
                return np.inf if q > 0.5 else -np.inf
            try:
                return float(np.quantile(a, q, method="nearest"))
            except TypeError:
                return float(np.quantile(a, q, interpolation="nearest"))

        upper = _q(mismatch_scores, 1 - self._mu)  # controls false-link rate
        lower = _q(match_scores, self._lambda)  # controls false-non-link rate
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
