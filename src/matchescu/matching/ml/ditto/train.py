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
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import (
    Traits,
    RecordExtraction,
    RecordIdAdapter,
    single_record,
)
from matchescu.csg import BinaryComparisonSpaceGenerator, BinaryComparisonSpaceEvaluator
from matchescu.matching.ml.ditto._ditto_dataset import DittoDataset
from matchescu.matching.ml.ditto._ditto_module import DittoModel
from matchescu.matching.ml.ditto._ditto_trainer import DittoTrainer
from matchescu.matching.ml.ditto._ditto_training_evaluator import DittoTrainingEvaluator
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReferenceIdentifier

DATADIR = os.path.abspath("data")
BERT_MODEL_NAME = "roberta-base"
LEFT_CSV_PATH = os.path.join(DATADIR, "abt-buy", "Abt.csv")
RIGHT_CSV_PATH = os.path.join(DATADIR, "abt-buy", "Buy.csv")
GROUND_TRUTH_PATH = os.path.join(DATADIR, "abt-buy", "abt_buy_perfectMapping.csv")

# set up abt extraction
abt_traits = list(
    Traits().int(["id"]).string(["name", "description"]).currency(["price"])
)
abt = CsvDataSource(LEFT_CSV_PATH, traits=abt_traits).read()
# set up buy extraction
buy_traits = list(
    Traits()
    .int(["id"])
    .string(["name", "description", "manufacturer"])
    .currency(["price"])
)
buy = CsvDataSource(RIGHT_CSV_PATH, traits=buy_traits).read()
# set up ground truth
gt = set(
    (
        EntityReferenceIdentifier(x[0], abt.name),
        EntityReferenceIdentifier(x[1], buy.name),
    )
    for x in pl.read_csv(
        os.path.join(DATADIR, "abt-buy", "abt_buy_perfectMapping.csv"),
        ignore_errors=True,
    ).iter_rows()
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


def _id(record, source):
    return EntityReferenceIdentifier(record[0], source)


def load_data_source(id_table: InMemoryIdTable, data_source: CsvDataSource) -> None:
    record_adapter = RecordIdAdapter(partial(_id, source=data_source.name))
    extract_references = RecordExtraction(data_source, record_adapter, single_record)
    for ref in extract_references():
        id_table.put(ref)


@contextmanager
def timer(start_message: str):
    logging.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    log.info("%s time elapsed: %.2f seconds", start_message, time_end - time_start)


@timer(start_message="train ditto")
def run_training():
    id_table = InMemoryIdTable()
    load_data_source(id_table, abt)
    load_data_source(id_table, buy)
    original_comparison_space_size = len(abt) * len(buy)
    comparison_space = create_comparison_space(
        id_table, gt, original_comparison_space_size
    )

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
    trainer = DittoTrainer(BERT_MODEL_NAME, Path(DATADIR).parent / "models", epochs=10)
    evaluator = DittoTrainingEvaluator(BERT_MODEL_NAME, xv, test)
    trainer.run_training(ditto, train, evaluator, True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    with warnings.catch_warnings(action="ignore"):
        run_training()
