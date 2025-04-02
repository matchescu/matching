import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import polars as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from matchescu.blocking import TfIdfBlocker
from matchescu.comparison_filtering import is_cross_source_comparison
from matchescu.csg import BinaryComparisonSpaceGenerator, BinaryComparisonSpaceEvaluator
from matchescu.data import Record
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import (
    Traits,
    RecordExtraction,
    single_record,
)
from matchescu.matching.evaluation.datasets import MagellanDataset
from matchescu.matching.ml.ditto._ditto_dataset import DittoDataset
from matchescu.matching.ml.ditto._ditto_module import DittoModel
from matchescu.matching.ml.ditto._ditto_trainer import DittoTrainer
from matchescu.matching.ml.ditto._ditto_training_evaluator import DittoTrainingEvaluator
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable, InMemoryIdTable
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
    EntityReferenceIdFactory,
)

log = logging.getLogger(__name__)


def create_comparison_space(id_table, ground_truth, initial_size):
    csg = (
        BinaryComparisonSpaceGenerator()
        .add_blocker(TfIdfBlocker(id_table, 0.23))
        .add_filter(is_cross_source_comparison)
    )
    comparison_space = csg()
    eval_cs = BinaryComparisonSpaceEvaluator(ground_truth, initial_size)
    metrics = eval_cs(comparison_space)
    print(metrics)
    return comparison_space


def _id(records: list[Record], source: str):
    return RefId(records[0][0], source)


def load_data_source(id_table: InMemoryIdTable, data_source: CsvDataSource) -> None:
    extract_references = RecordExtraction(
        data_source, partial(_id, source=data_source.name), single_record
    )
    for ref in extract_references():
        id_table.put(ref)


@contextmanager
def timer(start_message: str):
    logging.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    log.info("%s time elapsed: %.2f seconds", start_message, time_end - time_start)


def _extract_dataset(dataset_path: Path) -> tuple[IdTable, BinaryComparisonSpace, set]:
    abt_traits = list(Traits().string(["name", "description"]).currency(["price"]))
    abt = CsvDataSource(dataset_path / "Abt.csv", traits=abt_traits).read()
    buy_traits = list(
        Traits().string(["name", "description", "manufacturer"]).currency(["price"])
    )
    buy = CsvDataSource(dataset_path / "Buy.csv", traits=buy_traits).read()
    # set up ground truth
    gt_path = dataset_path / "abt_buy_perfectMapping.csv"
    gt = set(
        (RefId(row[0], abt.name), RefId(row[1], buy.name))
        for row in pl.read_csv(gt_path, ignore_errors=True).iter_rows()
    )

    id_table = InMemoryIdTable()
    load_data_source(id_table, abt)
    load_data_source(id_table, buy)
    original_comparison_space_size = len(abt) * len(buy)

    comparison_space = create_comparison_space(
        id_table, gt, original_comparison_space_size
    )

    return id_table, comparison_space, gt


@timer(start_message="load dataset")
def load_magellan_dataset(
    ds_path: Path,
    left_traits: Traits,
    left_id_factory: EntityReferenceIdFactory,
    right_traits: Traits | None = None,
    right_id_factory: EntityReferenceIdFactory | None = None,
) -> MagellanDataset:
    ds = MagellanDataset(ds_path)
    ds.load_left(left_traits, left_id_factory)
    ds.load_right(right_traits or left_traits, right_id_factory or left_id_factory)
    ds.load_splits()
    return ds


@timer(start_message="serialize+tokenize")
def get_magellan_data_loaders(
    model_name: str, magellan_ds: MagellanDataset, batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, xv_ds, test_ds = [
        DittoDataset(
            magellan_ds.id_table, split.comparison_space, split.ground_truth, tokenizer
        )
        for split in [
            magellan_ds.train_split,
            magellan_ds.valid_split,
            magellan_ds.test_split,
        ]
    ]
    return (
        train_ds.get_data_loader(batch_size, shuffle=True),
        train_ds.get_data_loader(batch_size * 16),
        train_ds.get_data_loader(batch_size * 16),
    )


@timer(start_message="train ditto")
def run_training(
    model_name: str,
    train: DataLoader,
    xv: DataLoader,
    test: DataLoader,
    model_dir: Path,
):
    ditto = DittoModel(model_name)
    trainer = DittoTrainer(model_name, model_dir, epochs=10)
    evaluator = DittoTrainingEvaluator(model_name, xv, test)
    trainer.run_training(ditto, train, evaluator, True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    dataset_names = [
        "amazon_google_exp_data",
    ]
    with warnings.catch_warnings(action="ignore"):
        for dataset_name in dataset_names:
            magellan_ds = load_magellan_dataset(
                Path(os.getcwd()) / "data" / "magellan" / dataset_name,
                Traits().string(["title", "manufacturer"]).currency(["price"]),
                lambda rows: RefId(rows[0]["id"], "tableA"),
                right_id_factory=lambda rows: RefId(rows[0]["id"], "tableB"),
            )
            ds_model_dir = Path(os.getcwd()) / "models" / dataset_name
            run_training(
                "roberta-base",
                *get_magellan_data_loaders("roberta-base", magellan_ds, 64),
                model_dir=ds_model_dir
            )
