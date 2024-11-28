import itertools

import polars as pl
import pytest


@pytest.fixture
def dataset_dir(data_dir):
    return data_dir / "cora"


@pytest.fixture
def cora(dataset_dir):
    return pl.read_csv(dataset_dir / "cora.csv", ignore_errors=True, has_header=False)



def test_cora_classes(data_dir, cora):
    groups = cora.group_by(pl.col("column_3"))
    gt = []
    group_count = 0
    for group_name, df in groups:
        ids = [row[0] for row in df.iter_rows()]
        gt.extend([
            {"id_left": row[0], "id_right": row[1]}
            for row in itertools.combinations(ids, 2)
        ])
        group_count += 1
    print("found", group_count, "groups")
    df = pl.DataFrame(gt)
    df.write_csv(data_dir / "cora_duplicates.csv", include_header=True)
