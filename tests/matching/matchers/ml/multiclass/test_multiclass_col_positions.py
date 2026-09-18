import pytest
from matchescu.reference_store.comparison_space import InMemoryComparisonSpace
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference, EntityReferenceIdentifier
from transformers import AutoTokenizer

from matchescu.matching.evaluation.data.splits import Split
from matchescu.matching.matchers.ml.multiclass.training import (
    AsymmetricMultiClassDataset,
)

_SERIALIZED_TEXT = "COL name VAL Acme COL city VAL NYC"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)


@pytest.fixture
def ref_pair():
    left = EntityReference(
        EntityReferenceIdentifier("a", "left"), {"name": "Acme", "city": "NYC"}
    )
    right = EntityReference(
        EntityReferenceIdentifier("b", "right"), {"name": "Acme", "city": "NYC"}
    )
    return left, right


@pytest.fixture
def dataset(ref_pair, tokenizer):
    left, right = ref_pair
    id_table = InMemoryIdTable()
    id_table.put(left).put(right)
    comparison_space = InMemoryComparisonSpace()
    comparison_space.put(left.id, right.id)
    split = Split(
        comparison_space=comparison_space,
        matcher_labels={(left.id, right.id): 1},
        gt_clusters={0: {left.id, right.id}},
    )
    return AsymmetricMultiClassDataset(id_table, split, tokenizer)


def _valid_positions(col_positions):
    return col_positions[col_positions >= 0].tolist()


def test_find_col_positions_detects_all_column_markers(dataset, tokenizer):
    input_ids = tokenizer(_SERIALIZED_TEXT, return_tensors="pt")["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    expected = [pos for pos, token in enumerate(tokens) if token == "col"]

    positions = dataset._find_col_positions(input_ids)

    assert _valid_positions(positions) == expected


def test_dataset_item_col_positions_not_sentinel(dataset):
    x_fwd, x_rev, _ = dataset[0]

    for item in [x_fwd, x_rev]:
        valid = _valid_positions(item["col_positions"])
        assert len(valid) >= 2
        assert len(valid) == item["col_positions"].numel()
