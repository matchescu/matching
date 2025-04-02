import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer

from matchescu.blocking import TfIdfBlocker
from matchescu.comparison_filtering import is_cross_source_comparison
from matchescu.data import Record
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import (
    Traits,
    RecordExtraction,
    single_record,
)
from matchescu.csg import BinaryComparisonSpaceGenerator, BinaryComparisonSpaceEvaluator
from matchescu.matching.ml.ditto._ditto_dataset import DittoDataset
from matchescu.matching.ml.ditto._ditto_module import DittoModel
from matchescu.matching.ml.ditto._ditto_trainer import DittoTrainer
from matchescu.matching.ml.ditto._ditto_training_evaluator import DittoTrainingEvaluator
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable, InMemoryIdTable
from matchescu.typing import EntityReferenceIdentifier as RefId

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


@timer(start_message="train ditto")
def run_training(dataset_path: Path, model_dir: Path):
    BERT_MODEL_NAME = "roberta-base"
    id_table, comparison_space, gt = _extract_dataset(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    ds = DittoDataset(
        comparison_space,
        id_table,
        gt,
        tokenizer,
        left_cols=("name", "description", "price"),
        right_cols=("name", "description", "manufacturer", "price"),
    )
    ditto = DittoModel(BERT_MODEL_NAME)
    train, xv, test = ds.split(3, 1, 1, 64)
    trainer = DittoTrainer(BERT_MODEL_NAME, model_dir, epochs=10)
    evaluator = DittoTrainingEvaluator(BERT_MODEL_NAME, xv, test)
    trainer.run_training(ditto, train, evaluator, True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    dataset_dir = Path(os.getcwd()) / "data"
    model_dir = Path(os.getcwd()) / "models"
    with warnings.catch_warnings(action="ignore"):
        run_training(dataset_dir / "abt-buy", model_dir)
