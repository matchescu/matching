import sys
import time
import warnings
import logging
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import click
import humanize
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from matchescu.data import Record
from matchescu.extraction import Traits
from matchescu.matching.evaluation.data import MagellanBenchmarkData, MagellanTraits
from matchescu.matching.matchers.ml.deepmatcher import DeepMatcherModule
from matchescu.matching.matchers.ml.deepmatcher.training._config import (
    TrainingConfig,
    DEFAULT_MODEL_DIR,
    DEFAULT_DATA_DIR,
)
from matchescu.matching.matchers.ml.deepmatcher.training._dataset import (
    DeepMatcherDataset,
)
from matchescu.matching.matchers.ml.deepmatcher.training._evaluator import (
    TrainingEvaluator,
)
from matchescu.matching.matchers.ml.deepmatcher.training._trainer import (
    DeepMatcherTrainer,
)
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("deepmatcher-training")


@contextmanager
def timer(start_message: str):
    log.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    duration = humanize.naturaldelta(timedelta(seconds=(time_end - time_start)))
    log.info("%s time elapsed: %s", start_message, duration)


@timer(start_message="load dataset")
def load_magellan_dataset(
    ds_path: Path, left_traits: Traits, right_traits: Traits | None = None
) -> MagellanBenchmarkData:
    ds = MagellanBenchmarkData(ds_path)
    ds.load_left(left_traits)
    ds.load_right(right_traits or left_traits)
    ds.load_splits()
    return ds


@timer(start_message="serialize+tokenize")
def get_datasets(
    magellan_ds: MagellanBenchmarkData,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DeepMatcherDataset, ...]:
    return tuple(
        DeepMatcherDataset(magellan_ds.id_table, split, tokenizer=tokenizer)
        for split in [
            magellan_ds.train_split,
            magellan_ds.valid_split,
            magellan_ds.test_split,
        ]
    )


@timer(start_message="train deepmatcher")
def train_on_benchmark_data(
    model_save_dir: Path,
    model_name: str,
    benchmark_data: MagellanBenchmarkData,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-5,
):
    vocab_size = len(tokenizer.get_vocab())
    train_ds, xv_ds, test_ds = get_datasets(benchmark_data, tokenizer)
    train, xv, test = (
        train_ds.get_data_loader(batch_size, shuffle=True),
        xv_ds.get_data_loader(batch_size * 16),
        test_ds.get_data_loader(batch_size * 16),
    )
    model = DeepMatcherModule(vocab_size, train_ds.attr_dims, train_ds.attr_count)
    dataset_logger = log.getChild(benchmark_data.name)
    trainer = DeepMatcherTrainer(
        model_name,
        model_save_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        logger=dataset_logger,
    )
    tb_log_dir = model_save_dir / model_name / "tensorboard"
    with TrainingEvaluator(
        model_name, xv, test, tb_log_dir, dataset_logger
    ) as evaluator:
        trainer.run_training(model, train, evaluator, True)


def table_a_id(rows: Iterable[Record]) -> RefId:
    return RefId(next(iter(rows))["id"], "tableA")


def table_b_id(rows: Iterable[Record]) -> RefId:
    return RefId(next(iter(rows))["id"], "tableB")


@click.command
@click.option(
    "-M",
    "--model-dir",
    "root_model_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_MODEL_DIR,
)
@click.option(
    "-D",
    "--dataset-dir",
    "root_data_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_DATA_DIR,
)
@click.option(
    "-f",
    "--config-file",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=DEFAULT_MODEL_DIR / "config.json",
)
def run_training(
    root_model_dir: Path,
    root_data_dir: Path,
    config_path: Path,
) -> None:
    root_model_dir = Path(root_model_dir)
    root_data_dir = Path(root_data_dir)
    benchmark_dataset_traits = MagellanTraits()
    config = TrainingConfig.load_json(config_path)
    model_name = "dm-hybrid-symmetric-attention"
    with warnings.catch_warnings(action="ignore"):
        for dataset_name in config.dataset_names:
            ds_path = root_data_dir / "magellan" / dataset_name
            ds_model_dir = root_model_dir / dataset_name
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            train_params = config.get(dataset=dataset_name)
            ds_traits = benchmark_dataset_traits[dataset_name]
            benchmark_data = load_magellan_dataset(ds_path, ds_traits)
            train_on_benchmark_data(
                ds_model_dir,
                model_name,
                benchmark_data,
                tokenizer,
                batch_size=train_params.batch_size,
                epochs=train_params.epochs,
                learning_rate=train_params.learning_rate,
            )


if __name__ == "__main__":
    run_training()
