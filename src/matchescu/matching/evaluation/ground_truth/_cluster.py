from collections import defaultdict
from pathlib import Path

import polars as pl
from matchescu.typing import EntityReferenceIdentifier as RefId


def read_csv(
    path: Path,
    has_header: bool = True,
    id_col: str | int = 0,
    source_col: str | int | None = None,
    label_col: str | int | None = None,
    source_name: str | None = None,
) -> dict[int, set[RefId]]:
    if source_name is None and source_col is None:
        raise ValueError("must specify at least one of source name or source column")
    df = pl.read_csv(path, has_header=has_header)
    if has_header:
        col_idx = {name: idx for idx, name in enumerate(df.columns)}
        id_col = col_idx.get(id_col, id_col)
        source_col = col_idx.get(source_col, source_col)
        label_col = col_idx.get(label_col, label_col)
    if label_col is None:
        label_col = -1
    clusters: dict[int, set[RefId]] = defaultdict(set)
    for row in df.iter_rows(named=False):
        cluster_id = row[label_col]
        source = row[source_col] if source_col is not None else source_name
        ref_id = RefId(label=row[id_col], source=source)
        clusters[cluster_id].add(ref_id)
    return clusters
