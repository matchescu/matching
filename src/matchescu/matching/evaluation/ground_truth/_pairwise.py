from pathlib import Path

import polars as pl
from matchescu.typing import EntityReferenceIdentifier as RefId


def read_csv(
    path: str | Path,
    *sources: str,
    has_header: bool = True,
    id_cols: tuple[str | int, str | int] | None = None,
    source_cols: tuple[str | int, str | int] | None = None,
    label_col: str | int | None = None,
) -> dict[tuple[RefId, RefId], int]:
    """Read a pairwise ground truth from a CSV file.

    A pairwise ground truth stores compared pair IDs sensitive to input order
    (it may contain both (x, y) and (y, x) comparisons) and their associated
    labels. The labels may be binary or not.

    The CSV file containing the ground truth is expected to contain at least
    the left and right hand side unique identifiers of the compared entity
    references. Optionally, the CSV file may contain the names of the data
    sources where the left and right hand side operands of the comparison are to
    be found. Additionally, if the CSV file also contains labels, the user can
    indicate the label column.

    At least one of ``source_cols`` or ``sources`` must be specified.

    :param path: CSV file containing pairwise ground truth.
    :param has_header: if this is true, the first record in the CSV file will be skipped.
    :param id_cols: customize the names of the CSV columns containing the IDs of
        the compared references.
    :param source_cols: optionally specify the names of the CSV columns
        containing the names of the data sources of each of the comparison
        operands.
    :param label_col: optional CSV column containing labels.
    :params sources: 1 or 2 data source names that override whatever is found in
        the ``source_cols``. Interpretation: all LHS use first source and, if
        specified, all RHS items use second source (otherwise, all items are
        assumed to be from the same source).
    """
    if len(sources) > 0 and source_cols is not None:
        raise ValueError("only one of 'sources' and 'source_cols' can be specified")
    if len(sources) == 0 and source_cols is None:
        raise ValueError("at least one of 'sources' or 'source_cols' must be specified")
    df = pl.read_csv(path, has_header=has_header)
    lid_col, rid_col = id_cols or (0, 1)
    lsrc_col, rsrc_col = source_cols or (None, None)
    if has_header:
        col_to_idx = {col: i for i, col in enumerate(df.columns)}
        lid_col = col_to_idx.get(lid_col, lid_col)
        rid_col = col_to_idx.get(rid_col, rid_col)
        lsrc_col = col_to_idx.get(lsrc_col, lsrc_col)
        rsrc_col = col_to_idx.get(rsrc_col, rsrc_col)
        label_col = col_to_idx.get(label_col, label_col)
    else:
        checked = lid_col, rid_col, lsrc_col, rsrc_col, label_col
        if any(val is not None and not isinstance(val, int) for val in checked):
            raise ValueError(
                "only int indexers supported for CSV files without header row"
            )

    def _get_ids(r):
        lsource = r[lsrc_col] if lsrc_col is not None else sources[0]
        rsource = (
            r[rsrc_col]
            if rsrc_col is not None
            else (sources[1] if len(sources) > 1 else lsource)
        )
        return RefId(r[lid_col], str(lsource)), RefId(r[rid_col], str(rsource))

    def _is_positive_label(r):
        if label_col is None:
            return True
        return r[label_col] > 0

    return {
        _get_ids(r): r[label_col] if label_col is not None else 1
        for r in df.iter_rows(named=False)
        if _is_positive_label(r)
    }
