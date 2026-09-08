import pytest
import torch
from matchescu.extraction import Traits
from transformers import AutoTokenizer

from matchescu.matching.evaluation.data.benchmark import MagellanBenchmarkData
from matchescu.matching.matchers.ml.deeper.training import DeepERDataset
from matchescu.matching.matchers.ml.deepmatcher.training import DeepMatcherDataset
from matchescu.matching.matchers.ml.ditto.training import DittoDataset
from matchescu.matching.matchers.ml.multiclass.training import (
    AsymmetricMultiClassDataset,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)


@pytest.fixture(scope="module")
def benchmark(data_dir):
    ag = data_dir / "amazon_google_exp_data"
    traits = Traits().string(["title", "manufacturer"]).currency(["price"])
    bd = MagellanBenchmarkData(ag)
    bd.load_left(traits).load_right(traits).load_splits()
    return bd


def _first_batch(loader):
    return next(iter(loader))


def test_deeper_dataset_labels_are_long(benchmark, tokenizer):
    ds = DeepERDataset(benchmark.id_table, benchmark.train_split, tokenizer)
    loader = ds.get_data_loader(batch_size=4)
    *_, labels = _first_batch(loader)
    assert labels.dtype == torch.long


def test_deepmatcher_dataset_labels_are_long(benchmark, tokenizer):
    ds = DeepMatcherDataset(benchmark.id_table, benchmark.train_split, tokenizer)
    loader = ds.get_data_loader(batch_size=4)
    batch = _first_batch(loader)
    assert batch["label"].dtype == torch.long


def test_ditto_dataset_labels_are_long(benchmark, tokenizer):
    ds = DittoDataset(benchmark.id_table, benchmark.train_split, tokenizer)
    loader = ds.get_data_loader(batch_size=4)
    *_, labels = _first_batch(loader)
    assert labels.dtype == torch.int64


def test_multiclass_dataset_labels_are_long(benchmark, tokenizer):
    ds = AsymmetricMultiClassDataset(
        benchmark.id_table, benchmark.train_split, tokenizer
    )
    loader = ds.get_data_loader(batch_size=4)
    *_, labels = _first_batch(loader)
    assert labels.dtype == torch.int64


def test_multiclass_dataset_item_has_col_positions(benchmark, tokenizer):
    ds = AsymmetricMultiClassDataset(
        benchmark.id_table, benchmark.train_split, tokenizer
    )
    x_fwd, x_rev, _ = ds[0]
    assert "col_positions" in x_fwd
    assert "col_positions" in x_rev
    assert x_fwd["col_positions"].dtype == torch.long
    assert x_rev["col_positions"].dtype == torch.long


def test_multiclass_dataset_batch_has_padded_col_positions(benchmark, tokenizer):
    ds = AsymmetricMultiClassDataset(
        benchmark.id_table, benchmark.train_split, tokenizer
    )
    loader = ds.get_data_loader(batch_size=4)
    x_fwd, x_rev, _ = _first_batch(loader)
    assert "col_positions" in x_fwd
    assert "col_positions" in x_rev
    batch_size = x_fwd["col_positions"].shape[0]
    assert batch_size == 4
    # padded dimension should be >= 1
    assert x_fwd["col_positions"].shape[1] >= 1
    # padding value is -1
    assert (x_fwd["col_positions"] >= -1).all()
