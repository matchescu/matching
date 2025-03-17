import logging
import os
import sys
import time
from contextlib import contextmanager

import polars as pl
from transformers import AutoTokenizer

from matchescu.matching.extraction import Traits, CsvDataSource
from matchescu.matching.blocking import BlockEngine
from matchescu.matching.ml.ditto._ditto_dataset import DittoDataset
from matchescu.matching.ml.ditto._ditto_module import DittoModel

DATADIR = os.path.abspath("data")
BERT_MODEL_NAME = "roberta-base"
LEFT_CSV_PATH = os.path.join(DATADIR, "abt-buy", "Abt.csv")
RIGHT_CSV_PATH = os.path.join(DATADIR, "abt-buy", "Buy.csv")
GROUND_TRUTH_PATH = os.path.join(DATADIR, "abt-buy", "abt_buy_perfectMapping.csv")

# set up abt extraction
abt_traits = list(Traits().int([0]).string([1, 2]).currency([3]))
abt = CsvDataSource(name="abt", traits=abt_traits).read_csv(LEFT_CSV_PATH)
# set up buy extraction
buy_traits = list(Traits().int([0]).string([1, 2, 3]).currency([4]))
buy = CsvDataSource(name="buy", traits=buy_traits).read_csv(RIGHT_CSV_PATH)
# set up ground truth
gt = set(
    pl.read_csv(
        os.path.join(DATADIR, "abt-buy", "abt_buy_perfectMapping.csv"),
        ignore_errors=True,
    ).iter_rows()
)

log = logging.getLogger(__name__)


def _id(row):
    return row[0]


def create_comparison_space():
    blocker = BlockEngine().add_source(abt, _id).add_source(buy, _id).tf_idf(0.25)
    blocker.filter_candidates_jaccard(0.6)
    blocker.update_candidate_pairs(False)
    metrics = blocker.calculate_metrics(gt)
    print(metrics)
    return blocker


@contextmanager
def timer(start_message: str):
    logging.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    log.info("%s time elapsed: %.2f seconds", start_message, time_end - time_start)


@timer(start_message="train ditto")
def run_training():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    ds = DittoDataset(
        create_comparison_space(),
        _id,
        _id,
        gt,
        tokenizer,
        left_cols=("name", "description", "price"),
        right_cols=("name", "description", "manufacturer", "price"),
    )
    ditto = DittoModel(BERT_MODEL_NAME)
    ditto.run_training(ds, BERT_MODEL_NAME, save_model=True, batch_size=128)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_training()